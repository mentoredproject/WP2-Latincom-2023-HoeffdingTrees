Code related to the research associated to the application of Hoeffding Tree Algorithms IoT DDoS Detection:

- Paper published at http://dx.doi.org/10.1109/LATINCOM59467.2023.10361862 (DE ARAÚJO JOSEPHIK, JOÃO GABRIEL ANDRADE ; SIQUEIRA, YAISSA ; MACHADO, KÉTLY GONÇALVES ; TERADA, ROUTO ; DOS SANTOS, ALDRI LUIZ ; NOGUEIRA, MICHELE ; BATISTA, DANIEL MACÊDO . Applying Hoeffding Tree Algorithms for Effective Stream Learning in IoT DDoS Detection. In: Proceedings of 2023 IEEE LatinAmerican Conference on Communications (LATINCOM), 2023)

# DDoS detection using Machine Learning
Repository of code related to paper "Applying Hoeffding Tree Algorithms for Effective
Stream Learning in IoT DDoS Detection", submitted to the IEEE Latincom 2023. Authors are:
- João Gabriel Andrade de Araujo Josephik
- Yaissa Siqueira
- Kétly Gonçalves Machado
- Routo Terada
- Aldri Luiz dos Santos
- Michele Nogueira
- Daniel Macedo Batista

Dependencies are:

> river==0.15.0

> numpy==1.24.2

> pandas==1.5.3


To replicate the experiments:

1. Download the TON_IoT "Processed Network" dataset, avaliable in <https://cloudstor.aarnet.edu.au/plus/s/ds5zW91vdgjEj9i?path=%2FProcessed_datasets%2FProcessed_Network_dataset>. Save the csv files at "ton/dataset/raw".
2. Run all the cells in "ton/dataset/dataset_selection.ipynb". This will create "ton/dataset/processed/train_data.csv", which will be used by the models.
3. Run all the cells in "ton/train_and_test.ipynb". This may take several hours. Data will be saved at stats/\<model_name\>/\<timestamp\>.csv and plots will be saved at plots/\<model_name\>/\<timestamp\>.csv.
4. To get the mean and standard deviation of the metrics, run the cells at "ton/results_analysis.ipynb".
