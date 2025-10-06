## How to solve performance issues

Let's start with a profile of the inference engine. 

![Alt text](/img/dp_2_tp_4_nsys_timeline_1.png)

This is the nsys profiling result of the rollout phase of a RL step. The inference configuration is the following. We are using a machine with 8 H100 GPU. On this machine, we have two inference engines running. Each inference engine instance uses tensor parallel size 4, i.e. TP = 4.

As you observed, the most obivous problem is that there're so many bubbles between each inference step. To understand the cause, we need to focus on one inference engine. But nsys didn't tell use which process belong to which inference engine. Luckily, we can reason using our knowledge about model parallelism. Since tensor parallelism do 2 all-reduce in each decode layers, the forward for each shard for that engine should end at approximately the same time. Let's pin those processses.

![Alt text](/img/dp_2_tp_4_nsys_timeline_2.png)

Let's take tp rank 3 as an example. 

Due to a plugin called `jekyll-titles-from-headings` which is supported by GitHub Pages by default. The above header (in the markdown file) will be automatically used as the pages title.

If the file does not start with a header, then the post title will be derived from the filename.

This is a sample blog post. You can talk about all sorts of fun things here.

---

### This is a header

#### Some T-SQL Code

```tsql
SELECT This, [Is], A, Code, Block -- Using SSMS style syntax highlighting
    , REVERSE('abc')
FROM dbo.SomeTable s
    CROSS JOIN dbo.OtherTable o;
```

#### Some PowerShell Code

```powershell
Write-Host "This is a powershell Code block";

# There are many other languages you can use, but the style has to be loaded first

ForEach ($thing in $things) {
    Write-Output "It highlights it using the GitHub style"
}
```
