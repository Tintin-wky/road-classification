# 常用指令

随机删除一定数量文件
```
find . -type f | shuf | head -n 10 | xargs -I {} rm {}
```

