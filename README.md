# Git版本控制

## 一、下载

下载地址：[Git - Downloads](https://git-scm.com/downloads)

全程默认即可

验证命令：`git -v`

## 二、新建git仓库

以github测试，新建一个github仓库命名为yolo11

## 二、使用

### 1. 初始化及常用指令

```shell
# 新建一个本地git仓库
git init

# 设置用户名与邮箱
git config --global user.name "reio"
git config --global user.email reiossss@163.com

# 检查当前用户全局config
git config --global -l

# 将需要进行版本控制的文件放入缓存区
git add .

# 查看所有文件状态
git status

# 将缓存区的文件提交到本地git仓库
git commit -m "first commit"

# 新建分支
git branch -M main

# 切换分支
git checkout main

# 将dev分支合并到当前分支
git merge dev

# 将reio设置为连接远程仓库路线的名称
git remote add reio https://github.com/reiossss/yolov10.git

# 修改远程仓库地址
git remote set-url reio https://github.com/reiossss/reio.git

# 将reio该条线路与远程仓库main分支关联
git push --set-upstream reio main

# 将reio该路线与远程仓库master分支检查更新
git pull reio master

# 提交文件到reio路线仓库内
git push

# 同步更新
git pull

```

### 2. 其他命令

```shell
# 列出所有本地分支
git branch

#列出所有远程分支
git branch -r

# 新建一个分支，但依旧停留在当前分支
git branch [branch-name]

# 新建一个分支，并切换到该分支
git checkout -b [branch]

# 合并指定分支到当前分支
git merge [branch]

# 删除分支
git branch -d [branch-name]

# 删除远程分支
git push origin --delete [branch-name]

# 删除远程分支
git branch -dr [remote/branch]

# 删除文件
git rm --cached 文件名  

```

### 3. 特殊命令

```shell
# 使用'-f'强行覆盖提交，加上'--set-upstream'进行允许合并
git push -f --set-upstream origin main

# 完全重置远程跟踪
git fetch --all --prune

# 从暂存区移除所有文件
git reset HEAD .

# 设置git代理，查看本机系统端口号，手动设置代理：设置 > 网咯和Internet > 代理
git config --global http.proxy 127.0.0.1:7897
git config --global https.proxy 127.0.0.1:7897

# 去除git代理
git config --global --unset http.proxy
git config --global --unset https.proxy

```
