# JDA 复现结果可视化 - 部署指南

## 1. Demo 地址放在哪里？

Demo 通过 **Vercel** 部署，部署后地址由 Vercel 自动生成，例如：
- `https://comp7404-group16-jda-reproduction-xxx.vercel.app`（或项目自定义域名）
- 在 [vercel.com](https://vercel.com) 登录后，进入项目 → **Settings** → **Domains** 可查看/配置访问地址

---

## 2. 从 finial_jda 迁移到 COMP7404-Group16

**原地址**：https://github.com/quziqi77777-lgtm/finial_jda/tree/main/jda-results-visualization  
**新地址**：https://github.com/caroline1677/COMP7404-Group16-JDA-Reproduction/tree/visualization/jda-results-visualization

### 已修改内容（本仓库已完成）

| 文件 | 修改 |
|------|------|
| `index.html` | GitHub 链接改为 `tree/visualization/jda-results-visualization` |
| `DEPLOY.md` | 部署说明更新为新仓库和 visualization 分支 |

### 迁移步骤（若你在本地从 finial_jda 迁出）

1. 将 `jda-results-visualization` 文件夹推送到 COMP7404-Group16 的 `visualization` 分支：

```bash
cd <你的 COMP7404-Group16 本地路径>
git checkout -b visualization   # 若已有该分支则 git checkout visualization
git add jda-results-visualization
git commit -m "Add JDA results visualization"
git push origin visualization
```

2. 在 Vercel 中：
   - 若已有项目：**Settings** → **Git** → 将 **Production Branch** 改为 `visualization`，**Root Directory** 设为 `jda-results-visualization`
   - 若新建项目：Import `COMP7404-Group16-JDA-Reproduction` → **Root Directory** 填 `jda-results-visualization`，**Branch** 选 `visualization` → Deploy

---

## 快速部署到 Vercel

### 方式一：从 COMP7404-Group16 仓库部署（推荐）

1. 确保 `jda-results-visualization` 在 [COMP7404-Group16-JDA-Reproduction](https://github.com/caroline1677/COMP7404-Group16-JDA-Reproduction) 的 `visualization` 分支下
2. 打开 [vercel.com](https://vercel.com)，用 GitHub 登录
3. 点击 **Add New Project** → 选择 `COMP7404-Group16-JDA-Reproduction`
4. **Root Directory** 设为 `jda-results-visualization`
5. **Branch** 选择 `visualization`（若使用该分支）
6. 点击 Deploy

### 方式二：新建独立仓库

1. 在 GitHub 上新建仓库
2. 本地执行：

```bash
cd jda-results-visualization
git init
git add .
git commit -m "Add JDA results visualization"
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

3. 在 Vercel 导入该仓库 → Deploy

---

## 部署失败：No python entrypoint found

若出现此错误，说明 Vercel 把整个仓库识别为 Python 项目。**必须**在项目设置中将 **Root Directory** 设为 `jda-results-visualization`。

---

## 可配置项

所有可配置项在 `index.html` 中：
- **标题/作者**：`<h1>` 和页脚
- **GitHub 链接**：About 页面的 `<a href="...">`
- **数据**：`allData` 对象

修改后重新 push，Vercel 会自动重新部署。

---

## 二维码

- 部署后访问页面，底部会显示当前网址的二维码
- 点击「下载二维码」可保存为 PNG
