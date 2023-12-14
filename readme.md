# MATH3801 数学规划 Course Project (Group 5)

## 文件管理方式: `Git` （以下来自 ChatGPT ）

使用 `Git` 可以通过命令行或使用图形用户界面（GUI）工具。我会先介绍基本的命令行用法：

### 1. **安装与配置 `Git`**：

在您的系统上安装Git。您可以从[Git官方网站](https://git-scm.com/)下载适合您操作系统的安装程序，并按照安装向导进行安装。

[参考文献](https://blog.csdn.net/mukes/article/details/115693833)

在安装 `Git` 后，首先需要配置用户信息：

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

这会将您的名字和邮箱关联到您的 `Git` 操作中，以便提交时进行标识。

### 2. **此仓库**：

如果您想要克隆（即复制）一个现有的Git仓库到本地，可以使用以下命令：

```bash
git clone <repository_url>
```

`<repository_url>`是本仓库的 URL，可在 `<> Code` 菜单中找到，有 HTML 和 SSH 两种路径。如果您有 SSH 密钥，建议使用 SSH 路径，否则使用 HTML 路径。

#### 2.1 SSH 密钥配置（推荐）

参考 [Github 官方文档](https://docs.github.com/zh/authentication/connecting-to-github-with-ssh)

### 3. **提交更改**：

VS Code 提供了较为完整的 `Git` 支持，您可以在左侧的 `Source Control` 栏中看到当前仓库的状态。在您对文件进行更改后，可以在 `Source Control` 栏中点击 `+` 按钮将更改添加到暂存区，然后在 `...` 按钮中选择 `Commit` 进行提交。在提交时，您需要填写提交信息，以便其他人了解您的更改。提交信息应当简明扼要，但又能够清晰地表达您的更改。
