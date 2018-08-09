

```python
import heapq

graph_C = {
'a': {'b': 3, 'c': 5, 'd': 4},
'b': {'d': 2},
'c': {'e': 4},
'd': {'c': 1, 'e': 4}
 }

def shortestPath(G, source, destination):
    queue,seen = [(0, source, [])], set()
    while True:
        (cost, v, path) = heapq.heappop(queue)
        if v not in seen:
            path = path + [v]
            seen.add(v)
            if v == destination:
                return cost, path
            for (next, c) in G[v].items():
                heapq.heappush(queue, (cost + c, next, path))

cost, path = shortestPath(graph_C, 'a', 'e')
print(cost, path)
```

    8 ['a', 'd', 'e']

