use crate::raw::build::BuilderNode;
use crate::raw::{CompiledAddr, NONE_ADDRESS};

#[derive(Debug)]
pub struct Registry {
    table: Vec<RegistryCell>,
    table_size: usize, // number of rows
    mru_size: usize,   // number of columns
}

#[derive(Debug)]
struct RegistryCache<'a> {
    cells: &'a mut [RegistryCell],
}

#[derive(Clone, Debug)]
pub struct RegistryCell {
    addr: CompiledAddr,
    node: BuilderNode,
}

#[derive(Debug)]
pub enum RegistryEntry<'a> {
    Found(CompiledAddr),
    NotFound(&'a mut RegistryCell),
    Rejected,
}

impl Registry {
    pub fn new(table_size: usize, mru_size: usize) -> Registry {
        let empty_cell = RegistryCell::none();
        let ncells = table_size.checked_mul(mru_size).unwrap();
        Registry { table: vec![empty_cell; ncells], table_size, mru_size }
    }

    pub fn entry<'a>(&'a mut self, node: &BuilderNode) -> RegistryEntry<'a> {
        if self.table.is_empty() {
            return RegistryEntry::Rejected;
        }
        let bucket = self.hash(node);
        let start = self.mru_size * bucket;
        let end = start + self.mru_size;
        RegistryCache { cells: &mut self.table[start..end] }.entry(node)
    }

    fn hash(&self, node: &BuilderNode) -> usize {
        // Basic FNV-1a hash as described:
        // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        //
        // In unscientific experiments, this provides the same compression
        // as `std::hash::SipHasher` but is much much faster.
        const FNV_PRIME: u64 = 1099511628211;
        let mut h = 14695981039346656037;
        h = (h ^ (node.is_final as u64)).wrapping_mul(FNV_PRIME);
        h = (h ^ node.final_output.value()).wrapping_mul(FNV_PRIME);
        for t in &node.trans {
            h = (h ^ (t.inp as u64)).wrapping_mul(FNV_PRIME);
            h = (h ^ t.out.value()).wrapping_mul(FNV_PRIME);
            h = (h ^ (t.addr as u64)).wrapping_mul(FNV_PRIME);
        }
        (h as usize) % self.table_size
    }
}

impl<'a> RegistryCache<'a> {
    fn entry(mut self, node: &BuilderNode) -> RegistryEntry<'a> {
        if self.cells.len() == 1 {
            let cell = &mut self.cells[0];
            if !cell.is_none() && &cell.node == node {
                RegistryEntry::Found(cell.addr)
            } else {
                cell.set_node(node);
                RegistryEntry::NotFound(cell)
            }
        } else if self.cells.len() == 2 {
            let cell1 = &mut self.cells[0];
            if !cell1.is_none() && &cell1.node == node {
                return RegistryEntry::Found(cell1.addr);
            }

            let cell2 = &mut self.cells[1];
            if !cell2.is_none() && &cell2.node == node {
                let addr = cell2.addr;
                self.cells.swap(0, 1);
                return RegistryEntry::Found(addr);
            }

            self.cells[1].set_node(node);
            self.cells.swap(0, 1);
            RegistryEntry::NotFound(&mut self.cells[0])
        } else {
            let find = |c: &RegistryCell| !c.is_none() && &c.node == node;
            if let Some(i) = self.cells.iter().position(find) {
                let addr = self.cells[i].addr;
                self.promote(i); // most recently used
                RegistryEntry::Found(addr)
            } else {
                let last = self.cells.len() - 1;
                self.cells[last].set_node(node); // discard LRU
                self.promote(last);
                RegistryEntry::NotFound(&mut self.cells[0])
            }
        }
    }

    fn promote(&mut self, mut i: usize) {
        assert!(i < self.cells.len());
        while i > 0 {
            self.cells.swap(i - 1, i);
            i -= 1;
        }
    }
}

impl RegistryCell {
    fn none() -> RegistryCell {
        RegistryCell { addr: NONE_ADDRESS, node: BuilderNode::default() }
    }

    fn is_none(&self) -> bool {
        self.addr == NONE_ADDRESS
    }

    pub fn insert(&mut self, addr: CompiledAddr) {
        self.addr = addr;
    }

    /// Overwrite this cell's node with a copy of `node`.
    ///
    /// The cell's transition buffer is reused, unless it is much bigger than
    /// `node` needs. Cells live as long as the builder, so a buffer that only
    /// ever grows ends up as big as the widest node that was ever stored in
    /// its cell. Over a long build, that tends towards room for 256 or more
    /// transitions in every cell (over 100MB for the default registry on
    /// 64-bit targets), even though most nodes have only a few transitions.
    ///
    /// `Vec` does not promise a growth strategy, but today a buffer grows to
    /// the larger of twice its old capacity and the length needed. So with a
    /// slack of twice what `node` needs (and never below 4), a buffer that
    /// was just grown to fit `node` is not shrunk again straight away. If
    /// that ever changes, the cost is an extra reallocation, not a wrong
    /// result.
    fn set_node(&mut self, node: &BuilderNode) {
        self.node.clone_from(node);
        let needed = self.node.trans.len().max(4);
        if self.node.trans.capacity() > 2 * needed {
            self.node.trans.shrink_to(needed);
        }
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;

    use super::{Registry, RegistryCache, RegistryCell, RegistryEntry};
    use crate::raw::build::BuilderNode;
    use crate::raw::{CompiledAddr, Output, Transition};

    fn node_with_trans(addr: CompiledAddr, len: usize) -> BuilderNode {
        BuilderNode {
            is_final: false,
            final_output: Output::zero(),
            trans: (0..len)
                .map(|i| Transition {
                    addr,
                    inp: i as u8,
                    out: Output::zero(),
                })
                .collect(),
        }
    }

    fn assert_rejected(entry: RegistryEntry) {
        match entry {
            RegistryEntry::Rejected => {}
            entry => panic!("expected rejected entry, got: {:?}", entry),
        }
    }

    fn assert_not_found(entry: RegistryEntry) {
        match entry {
            RegistryEntry::NotFound(_) => {}
            entry => panic!("expected nout found entry, got: {:?}", entry),
        }
    }

    fn assert_insert_and_found(reg: &mut Registry, bnode: &BuilderNode) {
        match reg.entry(&bnode) {
            RegistryEntry::NotFound(cell) => cell.insert(1234),
            entry => panic!("unexpected not found entry, got: {:?}", entry),
        }
        match reg.entry(&bnode) {
            RegistryEntry::Found(addr) => assert_eq!(addr, 1234),
            entry => panic!("unexpected found entry, got: {:?}", entry),
        }
    }

    #[test]
    fn empty_is_ok() {
        let mut reg = Registry::new(0, 0);
        let bnode = BuilderNode {
            is_final: false,
            final_output: Output::zero(),
            trans: vec![],
        };
        assert_rejected(reg.entry(&bnode));
    }

    #[test]
    fn one_final_is_ok() {
        let mut reg = Registry::new(1, 1);
        let bnode = BuilderNode {
            is_final: true,
            final_output: Output::zero(),
            trans: vec![],
        };
        assert_insert_and_found(&mut reg, &bnode);
    }

    #[test]
    fn one_with_trans_is_ok() {
        let mut reg = Registry::new(1, 1);
        let bnode = BuilderNode {
            is_final: false,
            final_output: Output::zero(),
            trans: vec![Transition {
                addr: 0,
                inp: b'a',
                out: Output::zero(),
            }],
        };
        assert_insert_and_found(&mut reg, &bnode);
        assert_not_found(
            reg.entry(&BuilderNode { is_final: true, ..bnode.clone() }),
        );
        assert_not_found(reg.entry(&BuilderNode {
            trans: vec![Transition {
                addr: 0,
                inp: b'b',
                out: Output::zero(),
            }],
            ..bnode.clone()
        }));
        assert_not_found(reg.entry(&BuilderNode {
            trans: vec![Transition {
                addr: 0,
                inp: b'a',
                out: Output::new(1),
            }],
            ..bnode.clone()
        }));
    }

    #[test]
    fn cache_works() {
        let mut reg = Registry::new(1, 1);

        let bnode1 = BuilderNode { is_final: true, ..BuilderNode::default() };
        assert_insert_and_found(&mut reg, &bnode1);

        let bnode2 =
            BuilderNode { final_output: Output::new(1), ..bnode1.clone() };
        assert_insert_and_found(&mut reg, &bnode2);
        assert_not_found(reg.entry(&bnode1));
    }

    // Each MRU size overwrites cells in its own branch of `entry`, so this
    // checks all three.
    #[test]
    fn overwritten_cell_releases_wide_capacity() {
        for mru_size in 1..=3 {
            let mut reg = Registry::new(1, mru_size);
            for addr in 0..mru_size {
                assert_insert_and_found(&mut reg, &node_with_trans(addr, 256));
            }
            for addr in 0..mru_size {
                assert_insert_and_found(&mut reg, &node_with_trans(addr, 1));
            }
            for cell in &reg.table {
                assert_eq!(cell.node.trans.len(), 1);
                assert!(
                    cell.node.trans.capacity() <= 8,
                    "mru_size {}: a 1-transition node kept capacity {}",
                    mru_size,
                    cell.node.trans.capacity(),
                );
            }
        }
    }

    #[test]
    fn prop_cell_capacity_tracks_current_node() {
        fn p(widths: Vec<u8>) -> bool {
            let mut reg = Registry::new(4, 2);
            for (addr, &width) in widths.iter().enumerate() {
                let node = node_with_trans(addr, width as usize + 1);
                if let RegistryEntry::NotFound(cell) = reg.entry(&node) {
                    cell.insert(addr);
                }
            }
            reg.table.iter().all(|cell| {
                let needed = cell.node.trans.len().max(4);
                cell.node.trans.capacity() <= 2 * needed
            })
        }
        quickcheck(p as fn(Vec<u8>) -> bool);
    }

    #[test]
    fn promote() {
        let bn = BuilderNode::default();
        let mut bnodes = vec![
            RegistryCell { addr: 1, node: bn.clone() },
            RegistryCell { addr: 2, node: bn.clone() },
            RegistryCell { addr: 3, node: bn.clone() },
            RegistryCell { addr: 4, node: bn.clone() },
        ];
        let mut cache = RegistryCache { cells: &mut bnodes };

        cache.promote(0);
        assert_eq!(cache.cells[0].addr, 1);
        assert_eq!(cache.cells[1].addr, 2);
        assert_eq!(cache.cells[2].addr, 3);
        assert_eq!(cache.cells[3].addr, 4);

        cache.promote(1);
        assert_eq!(cache.cells[0].addr, 2);
        assert_eq!(cache.cells[1].addr, 1);
        assert_eq!(cache.cells[2].addr, 3);
        assert_eq!(cache.cells[3].addr, 4);

        cache.promote(3);
        assert_eq!(cache.cells[0].addr, 4);
        assert_eq!(cache.cells[1].addr, 2);
        assert_eq!(cache.cells[2].addr, 1);
        assert_eq!(cache.cells[3].addr, 3);

        cache.promote(2);
        assert_eq!(cache.cells[0].addr, 1);
        assert_eq!(cache.cells[1].addr, 4);
        assert_eq!(cache.cells[2].addr, 2);
        assert_eq!(cache.cells[3].addr, 3);
    }
}
