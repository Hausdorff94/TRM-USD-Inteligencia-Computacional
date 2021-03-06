# TRM-USD-Inteligencia-Computacional

### Author:
Johny Vallejo-Sánchez

*javallejos@eafit.edu.co*

Universidad EAFIT

## RNN y redes LSTM

Todas las RNNs tienen la forma de una cadena de módulos repetitivos de red neuronal. En las RNN estándar, este módulo repetitivo tendrá una estructura simple, como una sola capa de *tanh*.

![](images/LSTM.jpg)

Las LSTM también son tipo cadena, pero el módulo de repetición tiene una estructura diferente. En lugar de tener una sola capa de red neuronal, hay cuatro que interactúan entre sí.

![](images/LSTM.png)

En el diagrama anterior, cada línea representa un vector completo, desde la salida de un nodo hasta las entradas de otros. Los círculos son operaciones puntuales, como la suma de vectores, los cuadros amarillos son capas de redes neuronales. Las de líneas juntas denotan concatenación, y las bifurcaciones denotan que su contenido se copia y las copias van a diferentes partes.

El estado de la celda es como una cinta transportadora. Corre port toda la cadena, con algunas interacciones lineales pequeñas. Es muy fácil que la información fluya sin cambios.


![](images/StatusCell.png)

El LSTM pueden eliminar o agregar información al estado de la célula, cuidadosamente regulado por estructuras llamadas puertas. Las puertas son una forma de dejar pasar la información. Se componen de una capa de red neuronal *sigmoide* y una operación de multiplicación puntual.

![](images/gate.png)

La salida de la capa *sigmoide* son números entre 0 y 1, que describe cuánto de cada componente que debe dejarse pasar. 0 significa "no dejar pasar nada", y 1 significa "dejar pasar todo". Un LSTM tiene tres de estas puertas, para proteger y controlar el estado de la celda.

El primer paso es decidir qué información vamos a tomar del estado de la celda. Esta decisión la determina una capa sigmoidea llamada "capa de puerta olvidada". Examina <img src="https://render.githubusercontent.com/render/math?math=h_t-1"> y <img src="https://render.githubusercontent.com/render/math?math=x_t">, y genera un número entre 0 y 1 para cada número en el estado de celda <img src="https://render.githubusercontent.com/render/math?math=C_t-1">.

![](images/FirstStep.png)

El siguiente paso es decidir qué nueva información vamos a almacenar en el estado de la celda. Esto tiene dos partes. 

1. Una capa *sigmoide* llamada "capa de puerta de entrada" decide qué valores actualizaremos. 
2. Una capa *tanh* crea un vector de nuevos valores candidatos, <img src="https://render.githubusercontent.com/render/math?math=\tilde{C}_t">, que podrían agregarse al estado. 
3. Se combinan los dos para crear una actualización del estado.

![](images/SecStep.png)

Ahora se actualiza el antiguo estado de la celda, <img src="https://render.githubusercontent.com/render/math?math=C_t-1">, en el nuevo estado de la celda <img src="https://render.githubusercontent.com/render/math?math=C_t">.

Multiplicamos el viejo estado por <img src="https://render.githubusercontent.com/render/math?math=f_t">, olvidando las valores que decidimos olvidar antes. Luego lo agregamos <img src="https://render.githubusercontent.com/render/math?math=i*\tilde{C}_t">. Estos son los nuevos valores candidatos, escalados según cuánto decidimos actualizar cada valor de estado.

![](images/ThStep.png)

Finalmente, se decide qué vamos a generar. Esta salida se basará en nuestro estado de celda, pero será una versión filtrada. 

1. Se ejecuta una capa *sigmoide* que decide qué partes del estado de la celda se va a generar. 
2. Se pone el estado de la celda en *tanh* (para que los valores estén entre -1 y 1) y lo multiplicamos por la salida de la puerta *sigmoide*, de modo que solo produzcamos las partes que decidimos.

![](images/finalStep.png)


**Información tomada de:** [Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)



