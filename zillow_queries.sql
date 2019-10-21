/* Use zillow database */
use zillow;

/* Find single family residential (propertylandusetypeid = 261) that were purchased between May and June of 2017 */
SELECT *
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE propertylandusetypeid = 261 and (transactiondate BETWEEN '2017-05-01' and '2017-06-30');


-- Find the distinct values of the counties.
SELECT DISTINCT(fips)
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE propertylandusetypeid = 261 and (transactiondate BETWEEN '2017-05-01' and '2017-06-30');

-- Find the tax rate distribution of fips = 6037 
SELECT taxamount/taxvaluedollarcnt as tax_rate
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE (propertylandusetypeid = 261) and (transactiondate BETWEEN '2017-05-01' and '2017-06-30') and fips = 6037;

-- Find the tax rate distribution of fips = 6059 
SELECT taxamount/taxvaluedollarcnt as tax_rate
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE (propertylandusetypeid = 261) and (transactiondate BETWEEN '2017-05-01' and '2017-06-30') and fips = 6059;


-- Find the tax rate distribution of fips = 6111 
SELECT taxamount/taxvaluedollarcnt as tax_rate
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE (propertylandusetypeid = 261) and (transactiondate BETWEEN '2017-05-01' and '2017-06-30') and fips = 6111;

-- Fields to bring into python df:

SELECT parcelid, bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, taxamount
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE (propertylandusetypeid = 261) and (transactiondate BETWEEN '2017-05-01' and '2017-06-30');

SELECT (bathroomcnt * calculatedfinishedsquarefeet) + (bedroomcnt*calculatedfinishedsquarefeet), taxvaluedollarcnt
from properties_2017
JOIN predictions_2017 using(parcelid)
WHERE (propertylandusetypeid = 261) and (transactiondate BETWEEN '2017-05-01' and '2017-06-30');


