use zillow;

SELECT DISTINCT(regionidzip)
FROM properties_2016;

SELECT *
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE unitcnt = 1.0 and (transactiondate BETWEEN '2017-05-01' and '2017-06-30');

SELECT taxamount/taxvaluedollarcnt as tax_rate, fips
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE (propertylandusetypeid = 261) and (transactiondate BETWEEN '2017-05-01' and '2017-06-30') and fips = 6037;

SELECT *
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE propertylandusetypeid = 261 and (transactiondate BETWEEN '2017-05-01' and '2017-06-30');

SELECT DISTINCT(fips)
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE propertylandusetypeid = 261 and (transactiondate BETWEEN '2017-05-01' and '2017-06-30');



