FROM mysql:8.0

# Copy initialization scripts and data
COPY init-db.sql /docker-entrypoint-initdb.d/01-init-db.sql
COPY models.csv /tmp/models.csv

# Configure MySQL for local data loading
RUN echo "[mysqld]" >> /etc/mysql/conf.d/mysql.cnf && \
    echo "local_infile=1" >> /etc/mysql/conf.d/mysql.cnf && \
    echo "secure_file_priv=" >> /etc/mysql/conf.d/mysql.cnf

# Expose MySQL port
EXPOSE 3306