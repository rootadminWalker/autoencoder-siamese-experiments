import cv2 as cv
import os

out = cv.VideoWriter('/tmp/project.avi', cv.VideoWriter_fourcc(*'DIVX'), 2, (640, 480))
cluster_plots_path = '/media/rootadminwalker/DATA/outputs/mnist_siamese_outputs/succeed/embedding_dim_2_ep100_loss_contrastive/cluster_plots'
cluster_plots = os.listdir(cluster_plots_path)
cluster_plots = sorted(cluster_plots, key=lambda c: int(c[2:5]))
last_frame = None
for cluster_plot in cluster_plots:
    plot = cv.imread(os.path.join(cluster_plots_path, cluster_plot))
    out.write(plot)
    last_frame = plot

for _ in range(4):
    out.write(last_frame)
out.release()
