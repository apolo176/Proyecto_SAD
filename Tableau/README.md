- [Evolución Anual de las reviews](#evolución-anual-de-las-reviews)
  - [Justificación del uso de la media bayesiana](#justificación-del-uso-de-la-media-bayesiana)
  - [Lectura del gráfico](#lectura-del-gráfico)
- [Comparativa demográfica ¿existe sesgo de género?](#comparativa-demográfica-existe-sesgo-de-género)
- [Evolución por países](#evolución-por-países)
- [Análisis Semántico; Puntos fuertes y débiles](#análisis-semántico-puntos-fuertes-y-débiles)
  - [Apple Music](#apple-music)
  - [Spotify](#spotify)
- [Clustering y tópicos; Análisis en profundidad](#clustering-y-tópicos-análisis-en-profundidad)

---

# Evolución Anual de las reviews
## Justificación del uso de la media bayesiana
Como cada año tiene una cantidad distinta de reviews y estas no están distribuidas de forma balanceada entre años es necesario tener en cuenta la cantidad de reviews anuales a la hora de representar la evolución de la puntuación. Por eso usamos la media bayesiana para generar la gráfica: https://en.wikipedia.org/wiki/Bayesian_average

M es la cantidad de reviews de las que nos "fiamos", los años con menos reviews que m son aplanados hacia la media global de todos los años para reducir el impacto de tener pocas reviews. (En la presentación se puede cambiar de forma interactiva)

## Lectura del gráfico
Ambas empresas muestran la misma tendencia, un par de años iniciales donde se mantienen con una buena nota media y un declive constante el resto de años.

Hay sobretodo cuatro franjas importantes en el gráfico:
- 2008-2014: Spotify no tiene competencia.
- 2015-2018: Aparece AppleMusic cuando Spotify está bajando por problemas de su plan gratuito.
- 2019-2022: Bajada estable de ambos de notas neutras a notas negativas.
- 2023-2025: Estancamiento en notas negativas y tendencia al uno.

Se entiende que hay un descontento general con ambas plataformas, y al sostenerse tanto en el tiempo el problema ha de ser estructural o de negocio más que un error o actualización puntual.
![Evolución Anual, Media Bayesiana](Fotos-Informe/1.png)

---

# Comparativa demográfica ¿existe sesgo de género?
Los hombres y las mujeres tienen la misma tendencia negativa a la hora de votar, por lo que concluímos que el descontento es universal y no existe sesgo de género.
![Evolución Anual, Media Bayesiana](Fotos-Informe/2.png)

---

# Evolución por países
Este gráfico muestra la tendencia de cada país sobre que empresa prefiere. Se grafican las superficies de las superficies en función de un coeficiente que se calcula de la siguiente forma:

|MediaSpotify|-|MediaAppleMusic|

Como la media de cada empresa puede ir  de \[0,5\] (0 es que no hay reviews, y el resto es la puntuación), el coeficiente irá de \[-5,5\], siendo -5 totalmente a favor de AppleMusic, 5 totalmente a facor de Spotify y 0 empate.

**Apunte**: Esta forma de medir a que empresa prefiere cada país no demuestra lo positivo o negativo de cada review, se representa igual un (5,3) que un (3,1), pero me parece que sigue siendo una buena representación por que no tiene sentido pintar España de un rojo intenso si AppleMusic tiene un 3, que es una nota neutra, aunque gane por un buen margen a Spotify.

El mapa es interactivo y se puede elegir que año mostrar de 2008 a 2025, además se puede ver una sección general que tiene en cuenta las reviews de todos los años (No la he subido por que gana Spotify por 0.2,0.15 en todos los países menos en Portugal que gana AppleMusic por -0.1, y un empate enseña bastante poco).

![Evolución Anual, Media Bayesiana](Fotos-Informe/3.png)
![Evolución Anual, Media Bayesiana](Fotos-Informe/4.png)


---

# Análisis Semántico; Puntos fuertes y débiles

## Apple Music

![Evolución Anual, Media Bayesiana](Fotos-Informe/5.png)

## Spotify

![Evolución Anual, Media Bayesiana](Fotos-Informe/6.png)


---

# Clustering y tópicos; Análisis en profundidad

![Evolución Anual, Media Bayesiana](Fotos-Informe/7.png)
![Evolución Anual, Media Bayesiana](Fotos-Informe/8.png)

---