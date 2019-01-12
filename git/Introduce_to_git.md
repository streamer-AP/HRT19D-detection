# Introduction to git and github

![git](https://i.imgur.com/dqRU6U3.jpg)

## 1.简介（Overview）

·Git is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency.
·Git is easy to learn and has a tiny footprint with lightning fast performance. It outclasses SCM tools like Subversion, CVS, Perforce, and ClearCase with features like cheap local branching, convenient staging areas, and multiple workflows.
·git不仅仅是个版本控制系统，它也是个内容管理系统(CMS),工作管理系统等。

## 2.版本控制系统(version control system)

### 2.1 版本控制系统是干什么的?

工程设计领域中使用版本控制管理工程蓝图的设计过程。 在 IT 开发过程中也可以使用版本控制思想管理代码的版本迭代.

### 2.2版本控制工具

1.集中式版本控制系统（Centralized Version Control System，CVCS)
CVS,SVN,VSS and so on
![集中式版本控制系统](https://i.imgur.com/6mVEGuL.png)
2.分布式版本控制系统（Distributed Version Control System，DVCS)
Git,Mercurial、 Bazaar、 Darcs and so on
![分布式版本控制系统](https://i.imgur.com/JJHmj2h.png)

## 3.Git的简介

### 3.1 Git的历史

![git的历史](https://i.imgur.com/9Uk2qAT.png)

### 3.2 官网

 [git的官网](https://git-scm.com/) ：https://git-scm.com/
 [github的官网](https://github.com/) ：https://github.com/
 [gitlab的官网](https://about.gitlab.com/) ：https://about.gitlab.com/

### 3.3 Git的优势

 1  大部分操作在本地完成，不需要联网
 2  完整性保证
 3  尽可能添加数据而不是删除或修改数据
 4  分支操作非常快捷流畅
 5  与 Linux 命令全面兼容

### 3.4 Git的安装

 自行安装

### 3.5 Git的结构

![git的结构](https://i.imgur.com/OQG2mZ2.png)

### 3.6 Git和代码托管中心

![代码托管中心](https://i.imgur.com/kQrUoTf.png)

这里简单说一下github和gitlab区别

* 外网环境下 GitHub 码云
* 局域网环境   GitLab服务器（相当于个人的github）

## 4.git操作

### 4.1本地库初始化

 首先我在自己的工作区创建一个目录gitDev，专门用来存放gitDev这个项目，将gitDev比作我将要开发的项目。

 `mkdir gitDev`

进入创建好的目录，pwd看下我的位置
~~接下来要开始表演了~~

### 4.1初始化

`git init`

![初始化](https://i.imgur.com/MiH11TI.png)

`ls -la`

![查看结果](https://i.imgur.com/9RrXvZU.png)

查看，多了.git的隐藏文件，说明已经初始化成功了

### 4.2设置签名

``` git
git config --global user.name xxxx
git config --global user.email xxxx
```

### 4.3 提交

`git statue`

![git状态](https://i.imgur.com/juQww5w.png)

创建一个文件 test
再看看

![创建文件](https://i.imgur.com/QpiUI8k.png)

`git add [file]`

![暂存文件](https://i.imgur.com/E7Zn6kP.png)

这只是存在于 暂存区 需要commit提交`

`git commit -m "comment about thsi commit"`

![提交](https://i.imgur.com/b3g6nkS.png)

### 4.4查看log

`git log --graph -- pretty="oneline"`

用最简短的图信息显示提交的历史.
`git reflog`

所有提交操作的参考记录.

### 4.5分支管理

![分支](https://i.imgur.com/ENlVKpf.png)

**分支可以理解多个功能同时推进，最后再合并在团队开发中有很大的作用**

好处：

1. 同时并行推进多个功能开发，提高开发效率
2. 各个分支在开发过程中，如果某一个分支开发失败，不会对其他分支有任何影响。失败的分支删除重新始即可。

下面是分支的基本操作：

1. 创建分支
  `git branch [name]`

2. 查看分支
  `git branch -v`

3. 切换分支
  `git checkout [name]`

  ![切换分支](https://i.imgur.com/NYpy7TQ.png)

## 5.github

### 5.1在github创建远程库

![创建仓库](https://i.imgur.com/6RkBzN6.png)

**远程库可选https协议或者ssh协议**

![选择协议](https://i.imgur.com/Era4Wlp.png)

### 5.2在本地添加远程库

`git remoteadd [name] [remote-address]`

### 5.3查看远程库

`git reomte -v`

![查看远程库](https://i.imgur.com/Ifa41cE.png)

### 5.4将本地库push到远程库中

`git push origin master`

![push到远程库](https://i.imgur.com/4SCTHtd.png)

在github中查看

![在git中查看](https://i.imgur.com/dEDOAnt.png)

### 5.5如果要下载已经有的库

`git clone`

![克隆](https://i.imgur.com/6dpNP2w.png)

## 6.总结

这里仅仅讲解了git和git最基本的概念和最基础的用法

想要更多的了解可以翻阅这本书

![推荐书籍](https://i.imgur.com/I0SU4wc.png)

推荐教程网站

官网：https://git-scm.com/book/en/v2

菜鸟教程-github: http://www.runoob.com/w3cnote/git-guide.html

菜鸟教程-git: http://www.runoob.com/git/git-tutorial.html

廖雪峰：https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000



**HRT_19D的团队协同将通过git和github来完成
git和github作为coder的开发利器 不可不掌握**


HRT_19D All Rights Reserved

