# make plots
python plot.py --in csv/train_acc.csv --out png/train_acc.png \
  -t "Training accuracy" \
  -x "Steps" \
  -y "Accuracy in %"

python plot.py --in csv/train_loss.csv --out png/train_loss.png \
  -t "Training loss" \
  -x "Steps" \
  -y "Loss"

python plot.py --in csv/train_loss_contrastive.csv --out png/train_loss_contrastive.png \
  -t "Training loss (only contrastive)" \
  -x "Steps" \
  -y "Contrastive loss"

python plot.py --in csv/train_loss_diversity.csv --out png/train_loss_diversity.png \
  -t "Training loss (only diversity)" \
  -x "Steps" \
  -y "Diversity loss"

python plot.py --in csv/train_loss_l2.csv --out png/train_loss_l2.png \
  -t "Training loss (only l2 penalty)" \
  -x "Steps" \
  -y "L2 penalty"

python plot.py --in csv/train_perplexity_cb1.csv --out png/train_perplexity_cb1.png \
  -t "Training perplexity of codebook 1" \
  -x "Steps" \
  -y "Perplexity"

python plot.py --in csv/train_perplexity_cb2.csv --out png/train_perplexity_cb2.png \
  -t "Training perplexity of codebook 2" \
  -x "Steps" \
  -y "Perplexity"


python plot.py --in csv/cb0_sim_avg.csv --out png/cb0_sim_avg.png \
  -t "Average similarity of codebook 1 on during training" \
  -x "Steps" \
  -y "Similarity"

python plot.py --in csv/cb0_sim_min.csv --out png/cb0_sim_min.png \
  -t "Minimum similarity of codebook 1 on during training" \
  -x "Steps" \
  -y "Similarity"

python plot.py --in csv/cb0_sim_max.csv --out png/cb0_sim_max.png \
  -t "Maximum similarity of codebook 1 on during training" \
  -x "Steps" \
  -y "Similarity"

python plot.py --in csv/cb1_sim_avg.csv --out png/cb1_sim_avg.png \
  -t "Average similarity of codebook 2 on during training" \
  -x "Steps" \
  -y "Similarity"

python plot.py --in csv/cb1_sim_min.csv --out png/cb1_sim_min.png \
  -t "Minimum similarity of codebook 2 on during training" \
  -x "Steps" \
  -y "Similarity"

python plot.py --in csv/cb1_sim_max.csv --out png/cb1_sim_max.png \
  -t "Maximum similarity of codebook 2 on during training" \
  -x "Steps" \
  -y "Similarity"


