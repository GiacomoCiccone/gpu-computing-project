# Progetto di GPU Computing
Il seguente progetto riguarda un'implementazione CUDA del primo libro di Peter Shirley, Ray Tracing in One Weekend.
Per compilare il progetto è necessario avere installato il toolkit CUDA e Make. E' necessario anche cambiare la prima riga del Makefile con la compute capability della propria GPU.
Per eseguire il progetto eseguire il comando `make run`, passando eventuali opzioni con la variabile `ARGS`. Per esempio, per eseguire il ray tracer con 1000 sample per pixel, lanciare il comando `make run ARGS="-n 1000"`. Gli argomenti validi sono i seguenti:

```bash
--width, -w <int>
--height, -h <int>
--max_depth, -d <int>
--num_samples, -n <int>
--use_gpu, -g
--output_file, -o <string>
```