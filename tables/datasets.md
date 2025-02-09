# Summary of datasets used for evaluating federated unlearning methods

| **Type** | **Task** | **Dataset** | **Source** | **# Records** | **# Features** | **# Classes** | **Paper** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Binary Data | Classification | **Adult** | Dheeru Dua and Casey Graff. 2017. UCI Machine Learning Repository. Retrieved from http://archive.ics.uci.edu/ml. | 48.842 | 14 | 2 |  |
|  |  | Cancer | “” | 699 | 10 | 2 |  |
|  |  | Diabetes | “” | 768 | 8 | 2 |  |
|  |  | Hepatitis | “” | 155 | 19 | 2 |  |
|  |  | German credit | “” | 1.000 | 20 | 2 |  |
|  |  | Hospital | Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. | 101.766 | 127 | 2 |  |
|  |  | UTKFace | Age progression/regression by conditional adversarial autoencoder | 20.705 | n.a. | 106 |  |
|  |  | US-Accident | Accident risk prediction based on heterogeneous sparse data: New dataset and insights | 3.000.000 | 30 | 3 |  |
|  |  | Foursquare | Participatory cultural mapping based on collective behavior data in location-based social networks | 528.878 | 446 | 30 |  |
|  |  | **Purchase-100** | Acquire Valued Shoppers Challenge. Retrieved from https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data. | 197.324 | 600 | 199 |  |
|  |  | **Texas-100** | Texas Hospital Inpatient Discharge Public Use Data File. Retrieved from https://www.dshs.texas.gov/thcic/hospit als/Inpatientpudf.shtm. | 67.330 | 6.170 | 100 |  |
| Image Data | Classification | Colored-MNIST | Invariant risk minimization | 70.000 | 28 x 28 x 1 | 2 |  |
|  |  | CH-MNIST | Multi-class texture analysis in colorectal cancer histology | 5.000 | 150 x 150 x 1 | 8 |  |
|  |  | SVHN | Reading digits in natural images with unsupervised feature learning. | 99.289 | 32 x 32 x 3 | 10 |  |
|  |  | Yale Face | Acquiring linear subspaces for face recognition under variable lighting | 2.414 | 168 x 192 x 1 | 38 |  |
|  |  | RCV1X | RCV1: A new benchmark collection for text categorization research | 800.000 | n.a. | 103 |  |
|  |  | Birds-200 | California Institute of Technology. CNS-TR-2010-001 | 11.788 | n.a. | 200 |  |
|  |  | FaceScrub | A data-driven approach to cleaning large face datasets | 100.000 | n.a. | 530 |  |
|  |  | ImageNet | ImageNet: A large-scale hierarchical image database | 1.281.167 | n.a. | 1.000 |  |
|  | Classification & Generation | IDC | Madabhushi. 2016. Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases. J. Pathol. Inf. 7 (2016), 29- | 277.524 | 50 x 50 x 3 | 2 |  |
|  |  | EyePACS | Kaggle. 2015. Diabetic Retinopathy Detection. Retrieved from [https://www.kaggle.com/c/diabetic-retinopathy-detection\\#references](https://www.kaggle.com/c/diabetic-retinopathy-detection%5C%5C#references). | 88.702 | n.a. | 5 |  |
|  |  | MNIST | Bengio, and Patrick Haffner. 1998. Gradient-based learning applied to document recognition. Proc. IEEE 86 | 70.000 | 28 x 28 x 1 | 10 |  |
|  |  | Fashion-MNIST | Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms | 70.000 | 28 x 28 x 1 | 10 |  |
|  |  | CIFAR-10 | Learning multiple layers of features from tiny images | 60.000 | 32 x 32 x 3 | 10 |  |
|  |  | CIFAR-100 | Learning multiple layers of features from tiny images | 60.000 | 32 x 32 x 3 | 100 |  |
|  |  | LFW | Labeled faces in the wild: A database for studying face recognition in unconstrained environments | 13.233 | 62 x 47 x 3 | 5.749 |  |
|  | Generation | CelebA | Deep learning face attributes in the wild. | 202.599 | 218 x 178 x 3 | 10.177 |  |
|  |  | MIMIC-III | MIMIC-III, a freely accessible critical care database | 46.520 | 1.071 | n.a. |  |
|  |  | Insta-NY | walk2friends: Inferring social links from mobility profiles. In CCS. ACM, 1943- | 34.336 | 4.048 | n.a. |  |
|  |  | ChestX-ray8 | ChestX-Ray8: Hospital-scale chest Xray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In CVPR. IEEE, 2097-2106. | 108.948 | 1024 x 1024 x 1 | 32.717 |  |
|  | Segmentation | Cityscapes | Schiele. 2016. The Cityscapes dataset for semantic urban scene understanding. In CVPR. IEEE. | 20.000 | n.a. | 30 |  |
|  |  | BDD100K | BDD100K: A diverse driving video database with scalable annotation tooling. arXiv preprint | 100.000 | n.a. | n.a. |  |
|  |  | Mapillary-Vistas | Kontschieder. 2017. The Mapillary
Vistas dataset for semantic understanding of street scenes. In | 25.000 | n.a. | 37 |  |
| Text Data | Classification | CSI | Daelemans. 2014. CLiPS stylometry investigation (CSI) corpus: A Dutch corpus for the detection of age, gender, personality, sentiment and deception in text. In LREC. 3081-3085. | 1.412 | n.a. | 2 |  |
|  |  | Review | 2013. Hidden factors and hidden topics: Understanding rating dimensions with review text. In | 364.038 | n.a. | 2 |  |
|  |  | Tweet EmoInt | Marquez. 2017. WASSA-2017 shared task on emotion intensity. In WASSA. | 7.097 | n.a. | 4 |  |
|  |  | Yelp-health | Shmatikov. 2019. Exploiting unintended feature leakage in collaborative learning. In S\&P. IEEE, | 17.938 | n.a. | 10 |  |
|  |  | News | Learning to filter netnews. In Machine Learning Proceedings 1995. Elsevier, | 20.000 | n.a. | 20 |  |
|  |  | Weibo | Weibo content corpus. In Proceedings of the [http://www.nlpir.org/wordpress/download/weibo_content_corpus.rar](http://www.nlpir.org/wordpress/download/w%C3%AAbal.content_corpus.rar) | 23.000 | n.a. | n.a. |  |
|  | Generation | Reddit comments | Reddit. 2017. Reddit comments dataset. Retrieved from https://bigquery.cloud.google.com/data set/fh-bigquery:redditcomments. | 83.293 | n.a. | n.a. |  |
|  |  | Dialogs | Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs | 220.579 | n.a. | n.a. |  |
|  |  | SATED | Extreme adaptation for personalized neural machine translation | 2.324 | n.a. | n.a. |  |
|  |  | WMT18 | Conference on Machine Translation(WMT'18). In WMT | n.a. | n.a. | n.a. |  |
|  | Embedding | Wikipedia | Large text compression benchmark. Retrieved from https://cs.fit.edu/mmahoney/compression/text.html | 150.000 | n.a. | n.a. |  |
|  |  | BookCorpus | Aligning books and movies: Towards story-like visual explanations by watching movies and reading books | 14.000 | n.a. | n.a. |  |
| Graph Data | Classification | Pubmed | Collective classification in network data | 19.717 | 500 | 3 |  |
|  |  | Citeseer | “” | 3.327 | 3.703 | 6 |  |
|  |  | Cora | “” | 2.708 | 1.433 | 7 |  |
|  |  | Lastfm | Characteristic functions on graphs: Birds of a feather, from statistical descriptors to parametric models | 7624 | 7.842 | 18 |  |
