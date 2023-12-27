from main import MFVI_with_subsampling_naive

mfvi = MFVI_with_subsampling_naive(
    model_dir='./models/LogisticRegression',
    dataset='sonar',
    observed_vars=['X', 'y']
)

mfvi.run(num_iters=1000)