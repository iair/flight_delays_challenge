
-- join the flyght table with the airport codes
DROP TABLE IF EXISTS flight_airport;
CREATE TABLE flight_airport AS
    (select *
     from "dataset_SCL" as f
         left join airports as a on f."Des-I"=a.icao_code);

select * from flight_airport;