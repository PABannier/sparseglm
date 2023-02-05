use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sparseglm::datasets::DenseDataset;
use sparseglm::estimators::{estimators::Lasso, traits::Fit};
use sparseglm::utils::helpers::compute_alpha_max;
use sparseglm::utils::test_helpers::generate_random_data;

fn bench_lasso(c: &mut Criterion) {
    let mut group = c.benchmark_group("lasso");
    group.sample_size(10);

    for n_samples in [10, 100] {
        for n_features in [100, 1000] {
            for reg in [0.1, 0.01, 0.005] {
                let (x, y) = generate_random_data(n_samples, n_features);

                let alpha_max = compute_alpha_max(x.view(), y.view());
                let alpha = alpha_max * reg;
                let dataset = DenseDataset::from((x, y));

                let clf = Lasso::params().alpha(alpha).verbose(false);
                let config = (n_samples, n_features, reg);
                let config_string = format!("{}, {}, {}", n_samples, n_features, reg);

                group.bench_with_input(
                    BenchmarkId::new("sparseglm", config_string),
                    &config,
                    |b, _| b.iter(|| clf.fit(&dataset).unwrap()),
                );
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_lasso);
criterion_main!(benches);
