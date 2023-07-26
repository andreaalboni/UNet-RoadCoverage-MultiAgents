# Road Extraction and Building Detection from Aerial Imagery 

L'obiettivo del progetto è creare e trainare due U-Net, CNN sviluppate per la segmentazione semantica, con lo scopo di, data un'immagine aerea, estrarre la rete 
stradale ed individuare gli edifici presenti.  

## Getting Started

Elenco e descrizione delle funzioni dei file presenti:
- Unet.py                                   -> creazione della rete neurale, architettura U-Net
- Train.py                                  -> train della rete neurale
- Test.py                                   -> test della rete neurale su un'immagine di qualsiasi dimensione
- smooth_predictions_by_blending_patches.py -> file richiesto dal test, migliora la precisione della predizione
- full_img_gmm.py                           -> nodo publisher ROS per casi in cui si analizza tutto l'ambiente
- img_select_gmm.py                         -> nodo publisher ROS per casi in cui si vuole selezionare una sola parte dell'ambiente
- Simulazione                               -> cartella contenente i launch file dei vari scenari per le simulazioni

### Prerequisiti

Dataset necessari per il training delle CNN:
- [Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
- [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset/code)

Requisiti software per il training:
- GPU con RAM >= 8Gb
- NVIDIA GPU Drivers and [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## Simulazioni

Text...

## Deployment

Link per il download dei modelli, in formato ".h5", pronti per l'utilizzo:
- [Road Extraction Model](https://drive.google.com/file/d/1dfdPuzAOjxv7tyFnCo3qPSDg3BL5kKfM/view?usp=sharing)
- [Buildings Detection Model](https://drive.google.com/file/d/15yyEJvJOZt-Vyrrf1LVo7sACHjIAR_w8/view?usp=sharing)

## Autore

  - **Andrea Alboni** 
