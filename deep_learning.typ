// ---------- basic style ----------  
#import "@preview/minimal-note:0.10.0": minimal-note

#show: minimal-note.with(
  title: [Notes on Deep Learning],
  author: [Aashay Kulkarni],
  date: datetime.today().display("[month repr:long], [year]")
)
// #set page(
//   paper: "a4", 
//   margin: 2.0cm,
//   header: [_Aashay Kulkarni's CUDA notes_ #line(length: 100%)], 
//   numbering: "1",
//   align: "justified",
// )  
// #set text(  
//   font: "Times New Roman",  // Ensure this font is installed or use "DejaVu Sans Mono"  
//   size: 10pt,  
// )  
// #set par(
//   justify: true,  
//   first-line-indent: 0pt,  
// )  
  
// ---------- notes ----------  
  
  

#pagebreak()

= Deep Feedforward Networks
\
They are also called feedforward neural networks or multilayer perceptrons (MLPs). The goal of a feedforward network is to approximate some function $f^*$. A feedforward network defines a mapping $y = f^*(x;theta)$ and learns the value of the parameters $theta$ that result in the best function approximation.

#parbreak()
Example: $f(x) = f^(3)(f^(2)(f^1(x)))$ is a neural network with 3 layers, with $f^1$ as the first layer. During training, we drive $f(x)$



