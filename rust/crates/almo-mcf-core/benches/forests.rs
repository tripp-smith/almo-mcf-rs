use criterion::{criterion_group, criterion_main, Criterion};

use almo_mcf_core::trees::forest::DynamicForest;
use almo_mcf_core::trees::shift::ShiftableForestCollection;
use almo_mcf_core::trees::LowStretchTree;

fn bench_forest_build(c: &mut Criterion) {
    let tails = vec![0, 1, 2, 3, 0, 2, 1, 3];
    let heads = vec![1, 2, 3, 4, 4, 4, 4, 2];
    let lengths = vec![1.0, 1.2, 0.9, 1.1, 1.4, 1.3, 0.8, 1.0];
    c.bench_function("forest_build", |b| {
        b.iter(|| {
            let tree = LowStretchTree::build_low_stretch(5, &tails, &heads, &lengths, 1).unwrap();
            let _forest = DynamicForest::new_from_tree(
                5,
                tails.clone(),
                heads.clone(),
                lengths.clone(),
                tree.tree_edges.clone(),
            )
            .unwrap();
        })
    });
}

fn bench_shift(c: &mut Criterion) {
    let tails = vec![0, 1, 2, 3, 0, 2, 1, 3];
    let heads = vec![1, 2, 3, 4, 4, 4, 4, 2];
    let lengths = vec![1.0, 1.2, 0.9, 1.1, 1.4, 1.3, 0.8, 1.0];
    c.bench_function("forest_shift", |b| {
        b.iter(|| {
            let mut collection =
                ShiftableForestCollection::build_deterministic(5, &tails, &heads, &lengths, 3, 2)
                    .unwrap();
            collection.mark_deleted(2);
            let _ = collection.shift(0);
        })
    });
}

criterion_group!(forests, bench_forest_build, bench_shift);
criterion_main!(forests);
