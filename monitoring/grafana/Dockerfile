FROM grafana/grafana-enterprise
COPY datasource.yml /etc/grafana/provisioning/datasources/local.yml
COPY dashboard.yml /etc/grafana/provisioning/dashboards/local.yml
COPY dashboard.json /var/lib/grafana/dashboards/visdrone.json
EXPOSE 3000