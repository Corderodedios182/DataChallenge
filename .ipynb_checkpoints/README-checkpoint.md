# DataChallenge

**Objetivo de este problema es determinar los factores que pueden llevar a los concesionarios a cumplir con los objetivos de satisfacción mínima.**

Datos *deIiveryDatasetChaIlenge.json* incluye datos de 29,389 rutas de camiones, realizadas por 29,389 empleados con las siguientes variables : 

**Primeras 10 columnas**

- **anonId:** Identificador único de la ruta
- **birthdate:** Fecha de nacimiento del empleado
- **routeDate:** Día en que se realizó la ruta entre el 20/05/11 y el 20/05/22
- **region:** Zona del mundo donde se ejecutó la ruta:
    - NA: North America
    - AMESA: Africa, Middle East and South Asia
    - APAC: Asia Pacific, Australia/New Zealand, and China 
    - LATAM: Latin America
    - Europe
- **gender:** Género autodeterminado del empleado {F-Female, M-Male, X-Non binary}
- **areaWealthLevel:** Desarrollo de la zona económica, en comparación con el conjunto del país (Low, Mid or High)
- **areaPopulation:** Población de la zona cubierta, en miles
- **badWeather:** Malas condiciones meteorológicas en la zona, como precipitaciones o viento fuerte
- **weatheRestrictions:** Afectaciones en la zona debido al clima
- **routeTotalDistance:** Distancia de la ruta recorrida en kms

**Segundo set de columnas**

- **numberOfShops:** Total Tiendas que cubrimos en la zona
- **marketShare:** Porcentaje de cuota de mercado que la empresa tiene en la zona en sus categorías.
- **avgAreaBenefits:** Beneficio económico semanal en la zona (en miles de $)
- **timeFromAvg:** Tiempo empleado en la ruta, comparado con la media (negativo significaría que se tardó menos que la media
- **advertising:** Inversión en material de punto de venta en las tiendas (de 0, que significa que no se invierte, a 3, que se invierte mucho)
- **emoloyeeLYScore:** Calificando la puntuación del año pasado (de 1 a 5, siendo 5 la más alta). Los nuevos empleados tienen 3 por defecto.
- **employeeTenure:** Tiempo que el empleado lleva en la empresa
  - 0: se han incorporado en los últimos 12 meses 
  - 1: de 1 a 3 años
  - 2: de 3 a 10 años
  - 3: mas de 1O años
- **emploIoyeePrevComps:** Número de empresas en las que el empleado trabajó anteriormente desarrollando la misma función (5 significa 3 o más).
- **success:** El distribuidor ha distribuido al menos el valor esperado (1) o menos (0)
