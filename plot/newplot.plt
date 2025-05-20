# set terminal postscript eps enhanced color 
set terminal pdf size 7, 3.5 

set format x '%.s%c'
set format y "%.3f"

set grid xtics ytics linetype 0 linewidth 1, linetype 0 linewidth 1
set style line 12 lc rgb "#808080" lt 0  # Dashed grid line style
set grid back ls 12

# set key outside right vertical
set rmargin 30
set key at screen 1, graph 1

# set yrange [ 5.0 : 9.5 ]
set yrange [0.00 : 30]
set xrange [0 : 4000000]


# Set the style for the plot
set style line 1 lc rgb "#E31A1C" pt 2 ps 1.5   # Red cross (pt 2)
set style line 2 lc rgb "#FF7F00" pt 5 ps 0.5 # Orange square (pt 5)
set style line 3 lc rgb "#33A02C" pt 7 ps 0.5  # Green circle (pt 7)
set style line 4 lc rgb "#1F78B4" pt 9 ps 1.5    # Red triangle (pt 9)
set style line 5 lc rgb "black" pt 1 ps 1.5

# Set labels
set xlabel "Parameters"
set ylabel "CER"
# set title "Plot"

# Plot the data
set output 'parzival.pdf'
plot 'parzival2.dat' using 1:(($3 == 0) ? $2 : 1/0) with points ls 3 title 'Clustering', \
     '' using 1:(($3 == 1) ? $2 : 1/0) with points ls 2 title 'Recursion', \
     '' using 1:($3==2?$2:1/0) with points ls 1 title 'Nothing', \
     'pareto_parzival2.dat' using 1:2 with lines ls 5 title 'Pareto frontier'

unset output

set output 'washington.pdf'
plot 'washington2.dat' using 1:(($3 == 0) ? $2 : 1/0) with points ls 3 title 'Clustering', \
     '' using 1:(($3 == 1) ? $2 : 1/0) with points ls 2 title 'Recursion', \
     '' using 1:($3==2?$2:1/0) with points ls 1 title 'Nothing', \
     'pareto_washington2.dat' using 1:2 with lines ls 5 title 'Pareto frontier'

unset output

set output 'saint_gall.pdf'
plot 'saint_gall2.dat' using 1:(($3 == 0) ? $2 : 1/0) with points ls 3 title 'Clustering', \
     '' using 1:(($3 == 1) ? $2 : 1/0) with points ls 2 title 'Recursion', \
     '' using 1:($3==2?$2:1/0) with points ls 1 title 'Nothing', \
     'pareto_saint_gall2.dat' using 1:2 with lines ls 5 title 'Pareto frontier'

unset output
