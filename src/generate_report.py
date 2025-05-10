import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark results
df = pd.read_csv('benchmark_results.csv')

# Plot insert times
plt.figure(figsize=(8,5))
for m in df['Method'].unique():
    sub = df[df['Method']==m]
    plt.plot(sub['N'], sub['Insert(s)'].astype(float), label=m)
plt.xlabel('N items')
plt.ylabel('Insert time (s)')
plt.title('Insert Performance')
plt.legend()
plt.savefig('insert_performance.png')

# Plot search times
plt.figure(figsize=(8,5))
for m in df['Method'].unique():
    sub = df[df['Method']==m]
    plt.plot(sub['N'], sub['Search(s)'].astype(float), label=m)
plt.xlabel('N items')
plt.ylabel('Search time (s)')
plt.title('Search Performance')
plt.legend()
plt.savefig('search_performance.png')

# Generate HTML report
template = f"""
<html>
<head><title>Hash Table Benchmark Report</title></head>
<body>
<h1>Benchmark Report</h1>
<h2>Insert Performance</h2>
<img src='insert_performance.png' alt='Insert Plot'>
<h2>Search Performance</h2>
<img src='search_performance.png' alt='Search Plot'>
</body>
</html>
"""
with open('report.html','w') as f:
    f.write(template)
print('Report generated: report.html')