## 数据生成

```
docker pull manimcommunity/manim:v0.19.0

docker run -dt --rm -v $(pwd):/app -w /app manimcommunity/manim:v0.19.0 bash

docker exec -it 9f10 bash
docker exec -it --user root 9f10 bash
```