# JDA 复现结果可视化 - 部署指南

## 快速部署到 Vercel

### 方式一：添加到现有仓库

1. 将 `jda-results-visualization` 文件夹推送到 [COMP7404-Group16-JDA-Reproduction](https://github.com/caroline1677/COMP7404-Group16-JDA-Reproduction)：

```bash
cd D:\Desktop\mdasc\7404\COMP7404-Group16-JDA-Reproduction-main
git add jda-results-visualization
git commit -m "Add JDA results visualization"
git push origin main
```

2. 打开 [vercel.com](https://vercel.com)，用 GitHub 登录
3. 点击 **Add New Project** → 选择 `COMP7404-Group16-JDA-Reproduction`
4. **Root Directory** 设为 `jda-results-visualization`，点击 Deploy
5. 部署完成后访问生成的网址，页面会显示二维码，可右键保存或点击「下载二维码」

### 方式二：新建独立仓库

1. 在 GitHub 上新建仓库 `jda-results`（或任意名称）
2. 本地执行：

```bash
cd D:\Desktop\mdasc\7404\COMP7404-Group16-JDA-Reproduction-main\jda-results-visualization
git init
git add .
git commit -m "Add JDA results visualization"
git remote add origin https://github.com/quziqi77777-lgtm/jda-results.git
git push -u origin main
```

3. 在 Vercel 导入该仓库 → Deploy

---

## 修改内容

所有可配置项在 `index.html` 中：

- **标题/作者**：`<h1>` 和 `.author`
- **GitHub 链接**：`config.repoUrl` 和 `config` 对象
- **数据**：`data` 对象中的 `paper` 和 `ours`

修改后重新 push，Vercel 会自动重新部署。

---

## 二维码

- 部署后访问页面，页面底部会显示当前网址的二维码
- 点击「下载二维码」可保存为 PNG
- 二维码内容即为当前页面 URL，扫码即可分享
