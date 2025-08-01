USER=$1
IP=$2

rsync \
    --rsync-path='mkdir -p /home/$USER/projects/clip-on-vision-assist-client && rsync' \
    -rv \
    --progress \
    ./ $USER@$IP:/home/$USER/projects/clip-on-vision-assist-client
