<?php


mkdir("ml_json_cats");

for ($i=0; $i < 10000; $i++) { 
	$khe = imagecreatefrompng("ml_small_cats/images/".$i.".png");

	for ($y=0; $y < 30; $y++) { 
		for ($x=0; $x < 30; $x++) { 

			$color = @imagecolorat($khe, $x, $y);
			$r = ($color >> 16) & 0xFF;
			$g = ($color >> 8) & 0xFF;
			$b = $color & 0xFF;
			$unitary_average = ($r + $b + $g)/(765);
			
			 //Makes the contrast a lil better thru sigmoid
			$unitary_average *= 1/(1+exp(-20*$unitary_average));

			$image_data_array[$x+$y*30] = $unitary_average;
		}
	}
	file_put_contents("ml_json_cats/".$i, json_encode($image_data_array));
}


?>