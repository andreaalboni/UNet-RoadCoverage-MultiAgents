# Road Extraction and Building Detection from Aerial Imagery 

L'obiettivo del progetto è creare e trainare due U-Net, CNN sviluppate per la segmentazione semantica, con lo scopo di, data un'immagine aerea, estrarre la rete 
stradale ed individuare gli edifici presenti.  

## Getting Started

Elenco e descrizione dei file presenti:
- Unet.py                                   -> creazione della rete neurale, architettura U-Net
- Train.py                                  -> train della rete neurale
- Test.py                                   -> test della rete neurale su un'immagine di qualsiasi dimensione
- smooth_predictions_by_blending_patches.py -> file richiesto dal test, migliora la precisione della predizione
- full_img_gmm.py                           -> nodo publisher ROS per casi in cui si analizza tutto l'ambiente
- img_select_gmm.py                         -> nodo publisher ROS per casi in cui si vuole selezionare una sola parte dell'ambiente
- Simulazioni                               -> cartella contenente i launch file dei vari scenari per le simulazioni

### Prerequisiti

Dataset necessari per il training delle CNN:
- [Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
- [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset/code)

Requisiti software per il training:
- GPU con RAM >= 8Gb
- NVIDIA GPU Drivers and [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## Simulazioni

Ai seguenti link sono disponibili per il download la cartella da inserire in ".../rotors_simulator/rotors_gazebo/models" e il file ".dae" di ciascuno degli scenari di simulazione:
- [Env - Quartiere di Reggio](https://drive.google.com/drive/folders/1L25QgqlFMfakWQTzxJSdDt4-lm1PjnNi?usp=sharing)
- [Env2 - Piazza del Colosseo](https://drive.google.com/drive/folders/1oCj5WPZFEup1hIQGeR_18bMNa0J6-P2m?usp=sharing)
- [Env4 - Strade Deserte](https://drive.google.com/drive/folders/13jji2yHSe3YBYaXJBXi-htyltGrIq8ox?usp=sharing)
 
Per avviare una simulazione, dopo aver correttamente collocato la cartella e il file ".dae" dello scenario scelto, procedere con il lancio, da terminale, del file ".launch" relativo all'ambiente scelto. Successivamente all'apertura di Gazebo, runnare uno dei due, full_img o img_select, tipo di studio dell'ambiente. Per ultimo, runnare il file "env(scelto)_coverage" per avviare il pilotaggio dei droni.

## Deployment

Link per il download dei modelli, in formato ".h5", pronti per l'utilizzo:
- [Road Extraction Model](https://drive.google.com/file/d/1dfdPuzAOjxv7tyFnCo3qPSDg3BL5kKfM/view?usp=sharing)
- [Buildings Detection Model](https://drive.google.com/file/d/15yyEJvJOZt-Vyrrf1LVo7sACHjIAR_w8/view?usp=sharing)

## Autore

  - **Andrea Alboni** 
