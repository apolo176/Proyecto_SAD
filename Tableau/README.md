- [Evolución Anual de las reviews](#evolución-anual-de-las-reviews)
  - [Justificación del uso de la media bayesiana](#justificación-del-uso-de-la-media-bayesiana)
  - [Lectura del gráfico](#lectura-del-gráfico)
- [Comparativa demográfica ¿existe sesgo de género?](#comparativa-demográfica-existe-sesgo-de-género)
- [Evolución por países](#evolución-por-países)
- [Análisis Semántico; Puntos fuertes y débiles](#análisis-semántico-puntos-fuertes-y-débiles)
  - [Apple Music](#apple-music)
  - [Spotify](#spotify)
  - [Conclusiones](#conclusiones)
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
El análisis semántico consiste en darle una polaridad a una palabra para analizar su impacto. La polaridad va de -1 a 1, siendo una palabra con polaridad -1 totalmente negativa y otra con 1 totalmente positiva. En estos gráficos también se tiene en cuenta la frecuencia de la palabra, cuanto más se menciona en las reviews más grande aparece en el gráfico.

Como hay muchas palabras (150 por defecto), he añadido un slide interactivo para elegir la cantidad de palabras y poder "hacer zoom" en las más importantes.
## Apple Music
En un primer vistazo se ve que tenemos más problemas que puntos fuertes:
- Puntos fuertes:
    - Calidad de audio: En el apartado de clustering se ve en detalle que es la característica principal que valoran los usuarios.
- Problemas:
    - Política de Apple y mala adaptación multiplataforma: La aplicación no funciona bien en dispositivos que no son de apple.
    - Bugs y problemas técnicos generales: Aparecen muchas palabras como service, access, update, account, sign, crashing...
    - Precios y problemas de facturación: Aparecen palabras como subscription, payment, money, trial... Apple Music no tiene una versión gratuita con funciones limitadas como Spotify, y muchas reviews se quejan de esta politica de empresa.

![Evolución Anual, Media Bayesiana](Fotos-Informe/5.png)

## Spotify
Spotify tiene una nube de palabras más neutras, aunque tiene muchas palabras completamente negativas se salva en más que en un solo aspecto:
- Puntos fuertes:
    - Accesibilidad y calidad general: devices, works, easy, experience, simple, platform... Estas palabras denotan que Spotify es una buena aplicación accesible y fácil de usar.
    - Recomendaciones y personalización: También se valora positivamente las recomendaciones de la aplicación y la capacidad de hacer mixes a medida.
- Problemas:
    - Suscripciones y Coste: La palabra más grande es claramente premium, aunque esta tiene una polaridad aproximada de -0.6, lo que indica que es mayoritariamente negativa pero hay usuarios que la valoran. Es decir, la mayoría de usuarios se quejan de las funciones que están reservadas a usuarios premium y dichos usuarios valoran positivamente las funciones que les facilita su plan.
    - Shuffle obligatorio y no poder saltar anuncios: Palabras como shuffle y skip demuestran que estas funciones son bastante molestas para los usuarios gratuitos.
    - Errores generales y actualizaciones: Crashing, update... Hay actualizaciones con problemas técnicos.

![Evolución Anual, Media Bayesiana](Fotos-Informe/6.png)

## Conclusiones
Ambos tienen problemas estructurales de negocio y problemas con las actualizaciones.
Mejoras propuestas:
- Apple Music:
    - Ser más flexible con el servicio multiplataforma y mejorar la experiencia de los dispositivos que no son de apple.
    - Implementar un plan gratuito con anuncios para reducir migraciones a Spotify.
- Spotify:
    - Implementar funcionalidades de audio sin pérdida para comerle terreno a Apple Music.
    - Reducir las restricciones de las cuentas gratuitas.

---

# Clustering y tópicos; Análisis en profundidad

![Evolución Anual, Media Bayesiana](Fotos-Informe/7.png)
![Evolución Anual, Media Bayesiana](Fotos-Informe/8.png)

---