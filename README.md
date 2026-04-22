# Nike 智能客服 Multi-Agent (RAG + 长期记忆)

## 功能
- Multi-agent 分工：导购推荐 / 售后 / 促销 / 库存与尺码
- RAG：基于 Nike 官方信息 + 产品知识库
- 长期记忆：对话偏好、尺码、预算等写入本地存储

## 运行环境
建议使用 conda：

```bash
conda create -n nike-agent python=3.10 -y
conda activate nike-agent
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

打开浏览器访问：
```
http://localhost:8000
```
