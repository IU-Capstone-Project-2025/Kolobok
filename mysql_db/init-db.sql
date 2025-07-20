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
