# Road Extraction and Building Detection from Aerial Imagery 

L'obiettivo del progetto è creare e trainare due U-Net, CNN sviluppate per la segmentazione semantica, con lo scopo di, data un'immagine aerea, estrarre la rete 
stradale ed individuare gli edifici presenti.  

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Prerequisiti

Dataset necessari per il training delle CNN:
- [Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
- [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset/code)

Requisiti software per il training:
- GPU con RAM >= 8Gb
- NVIDIA GPU Drivers and [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### Installazione

A step by step series of examples that tell you how to get a development
environment running

Say what the step will be

    Give the example

And repeat

    until finished

End with an example of getting some data out of the system or using it
for a little demo

## Running tests

Per testare il modello ottenuto, runnare il programma "test.py", il quale, dopo aver apportato oppurtune modifiche al percorso del modello, eseguirà una somma pesata sulle due predizioni 
sull'immagine selezionata, restituendo la definitiva. Una di queste predizioni sfrutta il metodo "blending-patches", contenuto nel file "smooth predictions by blending patches", mentre l'altra 
consiste nel predirre su ciascun patch, di 256x256 inn cui è scomposta l'immagine.

## Deployment

Link per il download dei modelli, in formato ".h5", pronti per l'utilizzo:
- [Road Extraction Model](https://drive.google.com/file/d/1dfdPuzAOjxv7tyFnCo3qPSDg3BL5kKfM/view?usp=sharing)
- [Buildings Detection Model](https://drive.google.com/file/d/15yyEJvJOZt-Vyrrf1LVo7sACHjIAR_w8/view?usp=sharing)

## Autore

  - **Andrea Alboni** 
