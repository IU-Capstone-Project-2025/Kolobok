CREATE TABLE IF NOT EXISTS models (
    id INT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id INT NOT NULL DEFAULT 0,
    INDEX idx_parent_id (parent_id),
    INDEX idx_name (name(255)),
    FULLTEXT INDEX ft_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

LOAD DATA INFILE '/tmp/models.csv' 
INTO TABLE models 
FIELDS TERMINATED BY ';' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n' 
IGNORE 1 ROWS 
(id, name, @parent_id_str)
SET parent_id = CAST(REPLACE(@parent_id_str, '"', '') AS UNSIGNED);

DELIMITER $$

DROP FUNCTION IF EXISTS IS_SIMILAR;
DROP FUNCTION IF EXISTS SIMILARITY_SCORE;
DROP FUNCTION IF EXISTS LEVENSHTEIN;

CREATE FUNCTION LEVENSHTEIN( s1 TEXT, s2 TEXT )
  RETURNS INT
  DETERMINISTIC
  BEGIN
    DECLARE s1_len, s2_len, i, j, c, c_temp, cost INT;
    DECLARE s1_char CHAR;
    DECLARE cv0, cv1 VARBINARY(256);

    SET s1_len = CHAR_LENGTH(s1), s2_len = CHAR_LENGTH(s2), cv1 = 0x00, j = 1, i = 1, c = 0;

    IF s1 = s2 THEN
      RETURN 0;
    ELSEIF s1_len = 0 THEN
      RETURN s2_len;
    ELSEIF s2_len = 0 THEN
      RETURN s1_len;
    ELSE
      WHILE j <= s2_len DO
        SET cv1 = CONCAT(cv1, UNHEX(HEX(j))), j = j + 1;
      END WHILE;
      WHILE i <= s1_len DO
        SET s1_char = SUBSTRING(s1, i, 1), c = i, cv0 = UNHEX(HEX(i)), j = 1;
        WHILE j <= s2_len DO
          SET c = c + 1;
          IF s1_char = SUBSTRING(s2, j, 1) THEN
            SET cost = 0; ELSE SET cost = 1;
          END IF;
          SET c_temp = CONV(HEX(SUBSTRING(cv1, j, 1)), 16, 10) + cost;
          IF c > c_temp THEN SET c = c_temp; END IF;
            SET c_temp = CONV(HEX(SUBSTRING(cv1, j+1, 1)), 16, 10) + 1;
            IF c > c_temp THEN
              SET c = c_temp;
            END IF;
            SET cv0 = CONCAT(cv0, UNHEX(HEX(c))), j = j + 1;
        END WHILE;
        SET cv1 = cv0, i = i + 1;
      END WHILE;
    END IF;
    RETURN c;
  END$$

CREATE FUNCTION SIMILARITY_SCORE(search_string TEXT, target_string TEXT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE s1_clean, s2_clean TEXT;
    DECLARE max_len INT;
    DECLARE lev_dist INT;
    DECLARE score FLOAT;

    SET s1_clean = LOWER(REGEXP_REPLACE(search_string, '[^a-zA-Z0-9]', ''));
    SET s2_clean = LOWER(REGEXP_REPLACE(target_string, '[^a-zA-Z0-9]', ''));

    IF CHAR_LENGTH(s1_clean) = 0 OR CHAR_LENGTH(s2_clean) = 0 THEN
        RETURN 0.0;
    END IF;

    SET lev_dist = LEVENSHTEIN(LEFT(s1_clean, 255), LEFT(s2_clean, 255));

    SET max_len = GREATEST(CHAR_LENGTH(s1_clean), CHAR_LENGTH(s2_clean));
    SET score = 1.0 - (lev_dist / max_len);
    
    IF LOWER(target_string) LIKE CONCAT(LOWER(search_string), '%') THEN
      SET score = score + 0.1;
    END IF;

    RETURN GREATEST(0.0, LEAST(1.0, score));
END$$


CREATE FUNCTION IS_SIMILAR(search_string TEXT, target_string TEXT, threshold FLOAT)
RETURNS BOOLEAN
DETERMINISTIC
BEGIN
    RETURN SIMILARITY_SCORE(search_string, target_string) >= threshold;
END$$

DELIMITER ;