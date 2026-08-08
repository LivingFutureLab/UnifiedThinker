import json

iflow_set_up_install_script = """
#!/bin/bash

apt-get update
apt-get install -y curl git wget xz-utils tar
apt-get install -y build-essential libc6-dev patch procps


wget --tries=10 --waitretry=2 http://nebula-cv-hz2.oss-cn-hangzhou.aliyuncs.com/user/eval/node-v22.18.0-linux-x64.tar.xz
tar -xf node-v22.18.0-linux-x64.tar.xz -C /opt/ && mv /opt/node-v22.18.0-linux-x64 /opt/nodejs
ln -sf /opt/nodejs/bin/node /usr/local/bin/node
ln -sf /opt/nodejs/bin/npm /usr/local/bin/npm
ln -sf /opt/nodejs/bin/npx /usr/local/bin/npx
ln -sf /opt/nodejs/bin/corepack /usr/local/bin/corepack

npm -v

# mkdir -p ~/.config
# wget --retry-connrefused --tries=10 --waitretry=2 -O ~/.config/iflow-cli.tgz 'https://cloud.iflow.cn/iflow-cli/iflow-ai-iflow-cli-roll.tgz'
# 
# npm i -g ~/.config/iflow-cli.tgz

echo "118.31.38.66  github.com" | tee -a /etc/hosts

npm i -g @iflow-ai/iflow-cli@{iflow_cli_version}


ln -s /opt/nodejs/bin/iflow /usr/local/bin/iflow

WORKING_DIR=$(pwd)
echo "工作目录: $WORKING_DIR"

mkdir /root/.iflow
"""


iflow_set_up_install_script_roll_zb = """
#!/bin/bash

# 检测系统类型
detect_system_and_version() {
    if [ -f /etc/debian_version ]; then
        . /etc/os-release
        if [ "$ID" = "ubuntu" ]; then
            echo "ubuntu:$VERSION_CODENAME"
        elif [ "$ID" = "debian" ]; then
            echo "debian:$VERSION_CODENAME"
        else
            echo "unknown:"
        fi
    else
        echo "unknown:"
    fi
}

# 设置阿里云APT源（使用 mirrors.cloud.aliyuncs.com）
setup_aliyun_apt_source() {
    SYSTEM_INFO=$(detect_system_and_version)
    SYSTEM=$(echo "$SYSTEM_INFO" | cut -d: -f1)
    CODENAME=$(echo "$SYSTEM_INFO" | cut -d: -f2)
    echo "系统类型: $SYSTEM, 版本代号: $CODENAME"
    
    # 备份原始源文件
    cp /etc/apt/sources.list /etc/apt/sources.list.backup
    
    if [ "$SYSTEM" = "debian" ]; then
        # 使用 mirrors.cloud.aliyuncs.com 源
        cat > /etc/apt/sources.list <<EOF
deb http://mirrors.cloud.aliyuncs.com/debian/ bookworm main non-free non-free-firmware contrib
deb http://mirrors.cloud.aliyuncs.com/debian-security/ bookworm-security main
deb http://mirrors.cloud.aliyuncs.com/debian/ bookworm-updates main non-free non-free-firmware contrib
EOF
    elif [ "$SYSTEM" = "ubuntu" ]; then
        # Ubuntu 使用 mirrors.cloud.aliyuncs.com 源
        if [ -z "$CODENAME" ]; then
            if [ -f /etc/os-release ]; then
                VERSION_ID=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
                case "$VERSION_ID" in
                    "24.04") CODENAME="noble" ;;
                    "22.04") CODENAME="jammy" ;;
                    "20.04") CODENAME="focal" ;;
                    *) CODENAME="noble" ;;
                esac
            else
                CODENAME="noble"
            fi
        fi
        cat > /etc/apt/sources.list <<EOF
deb http://mirrors.cloud.aliyuncs.com/ubuntu/ $CODENAME main restricted universe multiverse
deb http://mirrors.cloud.aliyuncs.com/ubuntu/ $CODENAME-security main restricted universe multiverse
deb http://mirrors.cloud.aliyuncs.com/ubuntu/ $CODENAME-updates main restricted universe multiverse
deb http://mirrors.cloud.aliyuncs.com/ubuntu/ $CODENAME-backports main restricted universe multiverse
EOF
    fi
    
    # 清理可能存在的其他源文件
    rm -f /etc/apt/sources.list.d/debian.sources
    
    # 设置APT配置，加速下载
    cat > /etc/apt/apt.conf.d/99speedup <<EOF
Acquire::http::Timeout "30";
Acquire::ftp::Timeout "30";
Acquire::Retries "3";
APT::Acquire::Retries "3";
APT::Get::Assume-Yes "true";
APT::Install-Recommends "false";
APT::Install-Suggests "false";
EOF
}

# 设置阿里云APT源
setup_aliyun_apt_source

# 清理APT缓存并更新
apt-get clean
rm -rf /var/lib/apt/lists/*
echo "阿里云APT源设置完成"

apt-get update
apt-get install -y curl git wget xz-utils tar
apt-get install -y build-essential libc6-dev patch procps


wget --tries=10 --waitretry=2 http://nebula-cv-hz2.oss-cn-hangzhou.aliyuncs.com/user/eval/node-v22.18.0-linux-x64.tar.xz
tar -xf node-v22.18.0-linux-x64.tar.xz -C /opt/ && mv /opt/node-v22.18.0-linux-x64 /opt/nodejs
ln -sf /opt/nodejs/bin/node /usr/local/bin/node
ln -sf /opt/nodejs/bin/npm /usr/local/bin/npm
ln -sf /opt/nodejs/bin/npx /usr/local/bin/npx
ln -sf /opt/nodejs/bin/corepack /usr/local/bin/corepack

npm -v

mkdir -p ~/.config
# wget --retry-connrefused --tries=10 'https://cloud.iflow.cn/iflow-cli/iflow-ai-iflow-cli-roll.tgz'
# mv iflow-ai-iflow-cli-roll.tgz ~/.config/iflow-cli.tgz

npm i -g https://cloud.iflow.cn/iflow-cli/iflow-ai-iflow-cli-roll.tgz

echo "118.31.38.66  github.com" | tee -a /etc/hosts

# npm i -g @iflow-ai/iflow-cli@{iflow_cli_version}


ln -s /opt/nodejs/bin/iflow /usr/local/bin/iflow

WORKING_DIR=$(pwd)
echo "工作目录: $WORKING_DIR"

mkdir /root/.iflow
"""




iflow_set_up_install_script_roll = """
#!/bin/bash

apt-get update
apt-get install -y curl git wget xz-utils tar
apt-get install -y build-essential libc6-dev patch procps


wget --tries=10 --waitretry=2 http://nebula-cv-hz2.oss-cn-hangzhou.aliyuncs.com/user/eval/node-v22.18.0-linux-x64.tar.xz
tar -xf node-v22.18.0-linux-x64.tar.xz -C /opt/ && mv /opt/node-v22.18.0-linux-x64 /opt/nodejs
ln -sf /opt/nodejs/bin/node /usr/local/bin/node
ln -sf /opt/nodejs/bin/npm /usr/local/bin/npm
ln -sf /opt/nodejs/bin/npx /usr/local/bin/npx
ln -sf /opt/nodejs/bin/corepack /usr/local/bin/corepack

npm -v

mkdir -p ~/.config
# wget --retry-connrefused --tries=10 'https://cloud.iflow.cn/iflow-cli/iflow-ai-iflow-cli-roll.tgz'
# mv iflow-ai-iflow-cli-roll.tgz ~/.config/iflow-cli.tgz

npm i -g https://cloud.iflow.cn/iflow-cli/iflow-ai-iflow-cli-roll.tgz

echo "118.31.38.66  github.com" | tee -a /etc/hosts

# npm i -g @iflow-ai/iflow-cli@{iflow_cli_version}


ln -s /opt/nodejs/bin/iflow /usr/local/bin/iflow

WORKING_DIR=$(pwd)
echo "工作目录: $WORKING_DIR"

mkdir /root/.iflow
"""



iflow_set_up_template = iflow_set_up_install_script + "\n" + """
#wget --retry-connrefused --tries=10 --waitretry=2 -O $WORKING_DIR/IFLOW.md 'http://nebula-cv-hz2.oss-cn-hangzhou.aliyuncs.com/user/eval/config/tb/IFLOW.md'
#wget --retry-connrefused --tries=10 --waitretry=2 -O $WORKING_DIR/IFLOW.md 'http://nebula-cv-hz2.oss-cn-hangzhou.aliyuncs.com/user/eval/config/IFLOW.md'
wget --retry-connrefused --tries=10 --waitretry=2 -O /root/.iflow/settings.json '{config}'

#cat IFLOW.md 
pwd
iflow -vx

"""


iflow_set_up_template = iflow_set_up_install_script_roll + "\n" + """
#wget --retry-connrefused --tries=10 --waitretry=2 -O $WORKING_DIR/IFLOW.md 'http://nebula-cv-hz2.oss-cn-hangzhou.aliyuncs.com/user/eval/config/tb/IFLOW.md'
#wget --retry-connrefused --tries=10 --waitretry=2 -O $WORKING_DIR/IFLOW.md 'http://nebula-cv-hz2.oss-cn-hangzhou.aliyuncs.com/user/eval/config/IFLOW.md'
wget --retry-connrefused --tries=10 --waitretry=2 -O /root/.iflow/settings.json '{config}'

#cat IFLOW.md 
pwd
iflow -vx
"""

def get_iflow_setting_template(iflow_selected_auth_type: str, ak_api_key, base_url, model, searchApiKey,telemetry_enable: bool=True, model_context_length:int=128000):
    config = {
        "selectedAuthType": iflow_selected_auth_type,
        "apiKey": ak_api_key,
        "baseUrl": base_url,
        "modelName": model,
        "searchApiKey": searchApiKey,
        "disableAutoUpdate": True,
        "shellTimeout": 360000,
        "tokensLimit": model_context_length
    }

    return json.dumps(config)

