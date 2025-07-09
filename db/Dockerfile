FROM mysql:8.0

COPY init-db.sql /docker-entrypoint-initdb.d/01-init-db.sql
COPY models.csv /tmp/models.csv

RUN echo "[mysqld]" >> /etc/mysql/conf.d/mysql.cnf && \
    echo "local_infile=1" >> /etc/mysql/conf.d/mysql.cnf && \
    echo "secure_file_priv=" >> /etc/mysql/conf.d/mysql.cnf

EXPOSE 3306
