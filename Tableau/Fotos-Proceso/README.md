# Explicaciones:
## Evolución Score:
Hay distintas versiones para poder elegir la que más guste y poder compararlas:
- EvolucionScore-1: Muestra la evolución anual de la media de las scores sin tener en cuenta cuantos comentarios hay en cada año. Los años con pocos comentarios pueden tener notas más extremas.
- EvolucionScore-2: Lo mismo que el 1 pero la línea es más gruesa o más fina en función de cuántos comentarios hay en ese año. Es bastante feo pero se entiende la relevancia de cada valor.
- EvolucionScore-m400: En vez de mostrar la media utiliza la media bayesiana, en esta se utiliza el parámetro 'm', el cual significa, más o menos: "a partir de cuantos comentarios anuales me fio del resultado". Los años con más de m comentarios se normalizan muy poco y los comentarios con menos tienden a la media global. No es lo mismo una media de 4.8 con 200 comentarios que una media de 4.6 con 1780 comentarios. https://en.wikipedia.org/wiki/Bayesian_average
- EvolucionScore-m600: Lo mismo que la anterior pero cambiando el m.
- CountScore: La cantidad de comentarios por año de cada empresa, para que valoreis lo de la media bayesiana y la m.
- Gender: La evolución anual de la media de las scores según el género. Decidme que opinais de las líneas grises, creo que así se nota mejor la diferencia pero quedan un poco feas. (aquí no incluyo la media bayesiana por que hay tan pocos comentarios de mujeres que rompe del todo la gráfica)
- GenderCount: La cantidad anual de comentarios por género para que veais la disparidad.
- GenderBarras: Lo mismo que Gender pero con un gráfico de barras, es más tosco pero se lee mejor.

