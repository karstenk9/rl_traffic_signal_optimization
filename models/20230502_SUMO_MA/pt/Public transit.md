
# Public transit module

By including some files in the simulation configuration, a somehow realistic public transit can be simulated.


## Usage

It is relatively simple to use the files that are already generated. Just include them in the corresponding elements of the `sumocfg` file. 

In our case, two files are to be included:
 - flows.rou.xml (route files)
 - stops.add.xml (additional files)

Both elements are child nodes of the `input` element.
For example:

    <configuration>
	    <input>
	        <net-file value="osm.net.xml"/>
	        <route-files value="osm.motorcycle.trips.xml,flows.rou.xml"/>
	        <additional-files value="osm.poly.xml,stops.add.xml"/>
	    </input>
    </configuration>

## Generate route file

With `ptlines_stops.xml` and `stops.add.xml`, which include information for the public transit lines and stops, one can use `ptlines2flows.py` to  generate a route file.

Run the following command:

`python "%SUMO_HOME%\tools\ptlines2flows.py" -n osm.net.xml -s stops.add.xml -l ptlines_stops.xml -o original_flows.rou.xml -p 600`

and a file named `original_flows.rou.xml` will be generated.

## Adjust route file

The automatically generated flows route file uses default departure time for all public transports. To align the simulation with reality, one can adjust the departure time for different lines. Here, `lines_begin.xlsx` contains such information we need. One can use `init_lines.py` to modify the begin time for each flow representing each bus or tram lines, and output the final file `flows.rou.xml`, 
which is what we need.

