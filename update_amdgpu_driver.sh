systemctl stop gdm3
rmmod amdgpu
# Old version 6.2.4-1664922.22.04
# New version 6.3.6-1718217.22.04

var=$1
if [ "$var" -eq 0 ]; then
echo "You have selected the classic driver" 
version=6.7.0-1756574.22.04_notModified
else
echo "You have selected the modified driver" 
version=6.7.0-1756574.22.04
fi


dkms remove -m amdgpu -v $version
#dkms remove -m amdgpu -v 6.2.4-1664922.22.04
dkms build -m amdgpu -v  $version
dkms install -m amdgpu -v  $version
insmod /lib/modules/6.2.0-35-generic/updates/dkms/amdgpu.ko 
check=$?

if [ "$check" -eq 0 ]; then
    echo "AMDGPU updated correctly"
else
    echo "AMDGPU NOT updated correct;"
fi

systemctl start gdm3