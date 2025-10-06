PROJECT_DIR="/home/ubuntu/poc_vanna_odoo"
VENV_DIR="$PROJECT_DIR/venv"
SOCKET_FILE="$PROJECT_DIR/vanna.sock"
SERVICE_NAME="vanna"
GUNICORN_MODULE="dev.poc_vanna:app"

# sudo tee > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
# [Unit]
# Description=Gunicorn instance to serve Vanna
# After=network.target

# [Service]
# User=root
# Group=www-data
# WorkingDirectory=$PROJECT_DIR
# ExecStartPre=find $PROJECT_DIR -type d -name "*-*-*-*-*" -exec rm -rf {} +
# ExecStartPre=find $PROJECT_DIR -type f -name "chroma.sqlite3" -exec rm -f {} \;
# ExecStart=$VENV_DIR/bin/gunicorn --workers 3 --bind unix:$SOCKET_FILE $GUNICORN_MODULE

# [Install]
# WantedBy=multi-user.target
# EOF


sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Gunicorn instance to serve Vanna
After=network.target

[Service]
User=root
Group=www-data
WorkingDirectory=$PROJECT_DIR
ExecStartPre=/usr/bin/find $PROJECT_DIR/models/hr -type d -name "*-*-*-*-*" -exec rm -rf {} +
ExecStartPre=/usr/bin/find $PROJECT_DIR/models/hr -type f -name "chroma.sqlite3*" -exec rm -f {} \;
ExecStart=$VENV_DIR/bin/gunicorn --workers 3 --bind unix:$SOCKET_FILE $GUNICORN_MODULE --access-logfile $PROJECT_DIR/access.log --error-logfile $PROJECT_DIR/error.log

[Install]
WantedBy=multi-user.target
EOF


sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}


# sudo cat > /etc/nginx/sites-available/${SERVICE_NAME} <<EOF
# server {
#     listen 9197;
#     server_name _;

#     location / {
#         proxy_pass http://unix:$SOCKET_FILE;
#         proxy_set_header Host \$host;
#         proxy_set_header X-Real-IP \$remote_addr;
#         proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
#         proxy_set_header X-Forwarded-Proto \$scheme;
#     }
# }
# EOF


sudo tee /etc/nginx/sites-available/${SERVICE_NAME} > /dev/null <<EOF
server {
    listen 9197;
    server_name _;

    location / {
        proxy_pass http://unix:$SOCKET_FILE;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/${SERVICE_NAME} /etc/nginx/sites-enabled/${SERVICE_NAME}

sudo nginx -t

sudo chown www-data:www-data /home/ubuntu/poc_vanna_odoo/vanna.sock
sudo chmod 660 /home/ubuntu/poc_vanna_odoo/vanna.sock

# show nginx logs
sudo tail -n 50 /var/log/nginx/error.log

curl --unix-socket /home/ubuntu/poc_vanna_odoo/vanna.sock http://localhost/models

# give necessay access to .sock file so it can be accessed by nginx
sudo chmod o+x /home
sudo chmod o+x /home/mypath

sudo chmod 766 /home/ubuntu/poc_vanna_odoo/vanna.sock
sudo chown root:www-data /home/ubuntu/poc_vanna_odoo/vanna.sock
