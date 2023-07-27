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
- print_pos.py                              -> consente di stampare sullo screenshot dell'ambiente la posizione finale dei droni, rappresentati da pallini verdi
- Simulazioni                               -> cartella contenente i launch file dei vari scenari per le simulazioni

### Prerequisiti

Dataset necessari per il training delle CNN:
- [Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
- [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset/code)

Requisiti software per il training:
- GPU con RAM >= 8Gb
- NVIDIA GPU Drivers and [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## Deployment

Link per il download dei modelli, in formato ".h5", pronti per l'utilizzo:
- [Road Extraction Model](https://drive.google.com/file/d/1dfdPuzAOjxv7tyFnCo3qPSDg3BL5kKfM/view?usp=sharing)
- [Buildings Detection Model](https://drive.google.com/file/d/15yyEJvJOZt-Vyrrf1LVo7sACHjIAR_w8/view?usp=sharing)

## Simulazioni

Ai seguenti link sono disponibili per il download la cartella da inserire in ".../rotors_simulator/rotors_gazebo/models" e il file ".dae" di ciascuno degli scenari di simulazione:
- [Env - Quartiere di Reggio](https://drive.google.com/drive/folders/1L25QgqlFMfakWQTzxJSdDt4-lm1PjnNi?usp=sharing)
- [Env2 - Piazza del Colosseo](https://drive.google.com/drive/folders/1oCj5WPZFEup1hIQGeR_18bMNa0J6-P2m?usp=sharing)
- [Env4 - Strade Deserte](https://drive.google.com/drive/folders/13jji2yHSe3YBYaXJBXi-htyltGrIq8ox?usp=sharing)
 
Per avviare una simulazione, dopo aver correttamente collocato la cartella e il file ".dae" dello scenario scelto, procedere con il lancio, da terminale, del file ".launch" relativo all'ambiente scelto. Successivamente, runnare uno delle due, full_img o img_select, tipologie di studio dell'ambiente, per intero o per selezione, decommentando le informazioni relative all'ambiente scelto. Per ultimo, runnare il file "env(scelto)_coverage" per avviare il pilotaggio dei droni. Runnare "print_pos" se si vuole visualizzare sull'immagine dell'ambiente dei pallini in corrispondenza della posizione finale dei droni.

## Autore

  - **Andrea Alboni** 
