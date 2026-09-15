from rlt_rsi.train import _seed_metrics


def test_generalization_gap_excludes_fitting_accuracy():
    run = {
        'train': {'accuracy': 1.0, 'n': 256, 'bce': 0.1},
        'dev': {'accuracy': 0.5, 'n': 128},
        'heldout': {'accuracy': 0.4, 'bce': 0.7},
        'final_train_bce': 0.1, 'seconds': 1.0,
    }
    first = _seed_metrics(run)
    run['train']['accuracy'] = 0.0
    second = _seed_metrics(run)
    assert first['in_distribution_accuracy'] == second['in_distribution_accuracy'] == 0.5
    assert first['length_generalization_drop'] == second['length_generalization_drop']
