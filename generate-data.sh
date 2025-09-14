# Data generation data for groupby and join

mkdir -p data
cd data/
Rscript ../db-benchmark/_data/join-datagen.R 1e7 0 0 0
Rscript ../db-benchmark/_data/join-datagen.R 2e7 0 0 0
Rscript ../db-benchmark/_data/join-datagen.R 4e7 0 0 0