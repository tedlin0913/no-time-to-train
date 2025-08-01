# All class splits for all datasets (centralised here)

voc_all_classes_1 = [ "aeroplane", "bicycle", "boat", "bottle", "car", "cat", "chair", "diningtable",
                     "dog", "horse", "person", "pottedplant", "sheep", "train", "tvmonitor", "bird",
                     "bus", "cow", "motorbike", "sofa"]
voc_all_classes_2 = ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable', 'dog',
                     'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor', 'aeroplane',
                     'bottle', 'cow', 'horse', 'sofa']
voc_all_classes_3 = ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable',
                     'dog', 'horse', 'person', 'pottedplant', 'train', 'tvmonitor', 'boat', 'cat',
                     'motorbike', 'sheep', 'sofa']

voc_split_1_seen_classes = [ "aeroplane", "bicycle", "boat", "bottle", "car", "cat", "chair", "diningtable",
                            "dog", "horse", "person", "pottedplant", "sheep", "train", "tvmonitor"]
voc_split_2_seen_classes = ["bicycle","bird","boat","bus","car","cat","chair","diningtable","dog","motorbike",
                            "person","pottedplant","sheep","train","tvmonitor"]
voc_split_3_seen_classes = ["aeroplane","bicycle","bird","bottle","bus","car","chair","cow","diningtable",
                            "dog","horse","person","pottedplant","train","tvmonitor"]



lvis = ['aerosol_can','air_conditioner','airplane','alarm_clock','alcohol','alligator','almond','ambulance','amplifier','anklet','antenna','apple','applesauce',
        'apricot','apron','aquarium','arctic_(type_of_shoe)','armband','armchair','armoire','armor','artichoke','trash_can','ashtray','asparagus','atomizer',
        'avocado','award','awning','ax','baboon','baby_buggy','basketball_backboard','backpack','handbag','suitcase','bagel','bagpipe','baguet','bait','ball',
        'ballet_skirt','balloon','bamboo','banana','Band_Aid','bandage','bandanna','banjo','banner','barbell','barge','barrel','barrette','barrow','baseball_base',
        'baseball','baseball_bat','baseball_cap','baseball_glove','basket','basketball','bass_horn','bat_(animal)','bath_mat','bath_towel','bathrobe','bathtub',
        'batter_(food)','battery','beachball','bead','bean_curd','beanbag','beanie','bear','bed','bedpan','bedspread','cow','beef_(food)','beeper','beer_bottle','beer_can',
        'beetle','bell','bell_pepper','belt','belt_buckle','bench','beret','bib','Bible','bicycle','visor','billboard','binder','binoculars','bird','birdfeeder','birdbath',
        'birdcage','birdhouse','birthday_cake','birthday_card','pirate_flag','black_sheep','blackberry','blackboard','blanket','blazer','blender','blimp','blinker','blouse',
        'blueberry','gameboard','boat','bob','bobbin','bobby_pin','boiled_egg','bolo_tie','deadbolt','bolt','bonnet','book','bookcase','booklet','bookmark','boom_microphone',
        'boot','bottle','bottle_opener','bouquet','bow_(weapon)','bow_(decorative_ribbons)','bow-tie','bowl','pipe_bowl','bowler_hat','bowling_ball','box','boxing_glove',
        'suspenders','bracelet','brass_plaque','brassiere','bread-bin','bread','breechcloth','bridal_gown','briefcase','broccoli','broach','broom','brownie','brussels_sprouts',
        'bubble_gum','bucket','horse_buggy','bull','bulldog','bulldozer','bullet_train','bulletin_board','bulletproof_vest','bullhorn','bun','bunk_bed','buoy','burrito',
        'bus_(vehicle)','business_card','butter','butterfly','button','cab_(taxi)','cabana','cabin_car','cabinet','locker','cake','calculator','calendar','calf','camcorder',
        'camel','camera','camera_lens','camper_(vehicle)','can','can_opener','candle','candle_holder','candy_bar','candy_cane','walking_cane','canister','canoe','cantaloup',
        'canteen','cap_(headwear)','bottle_cap','cape','cappuccino','car_(automobile)','railcar_(part_of_a_train)','elevator_car','car_battery','identity_card','card','cardigan',
        'cargo_ship','carnation','horse_carriage','carrot','tote_bag','cart','carton','cash_register','casserole','cassette','cast','cat','cauliflower','cayenne_(spice)',
        'CD_player','celery','cellular_telephone','chain_mail','chair','chaise_longue','chalice','chandelier','chap','checkbook','checkerboard','cherry','chessboard',
        'chicken_(animal)','chickpea','chili_(vegetable)','chime','chinaware','crisp_(potato_chip)','poker_chip','chocolate_bar','chocolate_cake','chocolate_milk','chocolate_mousse',
        'choker','chopping_board','chopstick','Christmas_tree','slide','cider','cigar_box','cigarette','cigarette_case','cistern','clarinet','clasp','cleansing_agent',
        'cleat_(for_securing_rope)','clementine','clip','clipboard','clippers_(for_plants)','cloak','clock','clock_tower','clothes_hamper','clothespin','clutch_bag','coaster','coat','coat_hanger',
        'coatrack','cock','cockroach','cocoa_(beverage)','coconut','coffee_maker','coffee_table','coffeepot','coil','coin','colander','coleslaw','coloring_material',
        'combination_lock','pacifier','comic_book','compass','computer_keyboard','condiment','cone','control','convertible_(automobile)','sofa_bed','cooker','cookie',
        'cooking_utensil','cooler_(for_food)','cork_(bottle_plug)','corkboard','corkscrew','edible_corn','cornbread','cornet','cornice','cornmeal','corset','costume',
        'cougar','coverall','cowbell','cowboy_hat','crab_(animal)','crabmeat','cracker','crape','crate','crayon','cream_pitcher','crescent_roll','crib','crock_pot',
        'crossbar','crouton','crow','crowbar','crown','crucifix','cruise_ship','police_cruiser','crumb','crutch','cub_(animal)','cube','cucumber','cufflink','cup',
        'trophy_cup','cupboard','cupcake','hair_curler','curling_iron','curtain','cushion','cylinder','cymbal','dagger','dalmatian','dartboard','date_(fruit)','deck_chair',
        'deer','dental_floss','desk','detergent','diaper','diary','die','dinghy','dining_table','tux','dish','dish_antenna','dishrag','dishtowel','dishwasher',
        'dishwasher_detergent','dispenser','diving_board','Dixie_cup','dog','dog_collar','doll','dollar','dollhouse','dolphin','domestic_ass','doorknob','doormat',
        'doughnut','dove','dragonfly','drawer','underdrawers','dress','dress_hat','dress_suit','dresser','drill','drone','dropper','drum_(musical_instrument)','drumstick',
        'duck','duckling','duct_tape','duffel_bag','dumbbell','dumpster','dustpan','eagle','earphone','earplug','earring','easel','eclair','eel','egg','egg_roll','egg_yolk',
        'eggbeater','eggplant','electric_chair','refrigerator','elephant','elk','envelope','eraser','escargot','eyepatch','falcon','fan','faucet','fedora','ferret',
        'Ferris_wheel','ferry','fig_(fruit)','fighter_jet','figurine','file_cabinet','file_(tool)','fire_alarm','fire_engine','fire_extinguisher','fire_hose','fireplace',
        'fireplug','first-aid_kit','fish','fish_(food)','fishbowl','fishing_rod','flag','flagpole','flamingo','flannel','flap','flash','flashlight','fleece','flip-flop_(sandal)',
        'flipper_(footwear)','flower_arrangement','flute_glass','foal','folding_chair','food_processor','football_(American)','football_helmet','footstool','fork','forklift',
        'freight_car','French_toast','freshener','frisbee','frog','fruit_juice','frying_pan','fudge','funnel','futon','gag','garbage','garbage_truck','garden_hose','gargle',
        'gargoyle','garlic','gasmask','gazelle','gelatin','gemstone','generator','giant_panda','gift_wrap','ginger','giraffe','cincture','glass_(drink_container)','globe',
        'glove','goat','goggles','goldfish','golf_club','golfcart','gondola_(boat)','goose','gorilla','gourd','grape','grater','gravestone','gravy_boat','green_bean',
        'green_onion','griddle','grill','grits','grizzly','grocery_bag','guitar','gull','gun','hairbrush','hairnet','hairpin','halter_top','ham','hamburger','hammer',
        'hammock','hamper','hamster','hair_dryer','hand_glass','hand_towel','handcart','handcuff','handkerchief','handle','handsaw','hardback_book','harmonium','hat',
        'hatbox','veil','headband','headboard','headlight','headscarf','headset','headstall_(for_horses)','heart','heater','helicopter','helmet','heron','highchair',
        'hinge','hippopotamus','hockey_stick','hog','home_plate_(baseball)','honey','fume_hood','hook','hookah','hornet','horse','hose','hot-air_balloon','hotplate',
        'hot_sauce','hourglass','houseboat','hummingbird','hummus','polar_bear','icecream','popsicle','ice_maker','ice_pack','ice_skate','igniter','inhaler','iPod',
        'iron_(for_clothing)','ironing_board','jacket','jam','jar','jean','jeep','jelly_bean','jersey','jet_plane','jewel','jewelry','joystick','jumpsuit','kayak','keg',
        'kennel','kettle','key','keycard','kilt','kimono','kitchen_sink','kitchen_table','kite','kitten','kiwi_fruit','knee_pad','knife','knitting_needle','knob',
        'knocker_(on_a_door)','koala','lab_coat','ladder','ladle','ladybug','lamb_(animal)','lamb-chop','lamp','lamppost','lampshade','lantern','lanyard','laptop_computer',
        'lasagna','latch','lawn_mower','leather','legging_(clothing)','Lego','legume','lemon','lemonade','lettuce','license_plate','life_buoy','life_jacket','lightbulb',
        'lightning_rod','lime','limousine','lion','lip_balm','liquor','lizard','log','lollipop','speaker_(stero_equipment)','loveseat','machine_gun',
        'magazine','magnet','mail_slot','mailbox_(at_home)','mallard','mallet','mammoth','manatee','mandarin_orange','manger','manhole','map','marker','martini',
        'mascot','mashed_potato','masher','mask','mast','mat_(gym_equipment)','matchbox','mattress','measuring_cup','measuring_stick','meatball','medicine','melon',
        'microphone','microscope','microwave_oven','milestone','milk','milk_can','milkshake','minivan','mint_candy','mirror','mitten','mixer_(kitchen_tool)','money',
        'monitor_(computer_equipment) computer_monitor','monkey','motor','motor_scooter','motor_vehicle','motorcycle','mound_(baseball)','mouse_(computer_equipment)',
        'mousepad','muffin','mug','mushroom','music_stool','musical_instrument','nailfile','napkin','neckerchief','necklace','necktie','needle','nest','newspaper',
        'newsstand','nightshirt','nosebag_(for_animals)','noseband_(for_animals)','notebook','notepad','nut','nutcracker','oar','octopus_(food)','octopus_(animal)',
        'oil_lamp','olive_oil','omelet','onion','orange_(fruit)','orange_juice','ostrich','ottoman','oven','overalls_(clothing)','owl','packet','inkpad','pad','paddle',
        'padlock','paintbrush','painting','pajamas','palette','pan_(for_cooking)','pan_(metal_container)','pancake','pantyhose','papaya','paper_plate','paper_towel',
        'paperback_book','paperweight','parachute','parakeet','parasail_(sports)','parasol','parchment','parka','parking_meter','parrot','passenger_car_(part_of_a_train)',
        'passenger_ship','passport','pastry','patty_(food)','pea_(food)','peach','peanut_butter','pear','peeler_(tool_for_fruit_and_vegetables)','wooden_leg','pegboard',
        'pelican','pen','pencil','pencil_box','pencil_sharpener','pendulum','penguin','pennant','penny_(coin)','pepper','pepper_mill','perfume','persimmon','person','pet',
        'pew_(church_bench)','phonebook','phonograph_record','piano','pickle','pickup_truck','pie','pigeon','piggy_bank','pillow','pin_(non_jewelry)','pineapple','pinecone',
        'ping-pong_ball','pinwheel','tobacco_pipe','pipe','pistol','pita_(bread)','pitcher_(vessel_for_liquid)','pitchfork','pizza','place_mat','plate','platter','playpen',
        'pliers','plow_(farm_equipment)','plume','pocket_watch','pocketknife','poker_(fire_stirring_tool)','pole','polo_shirt','poncho','pony','pool_table','pop_(soda)',
        'postbox_(public)','postcard','poster','pot','flowerpot','potato','potholder','pottery','pouch','power_shovel','prawn','pretzel','printer','projectile_(weapon)',
        'projector','propeller','prune','pudding','puffer_(fish)','puffin','pug-dog','pumpkin','puncher','puppet','puppy','quesadilla','quiche','quilt','rabbit','race_car',
        'racket','radar','radiator','radio_receiver','radish','raft','rag_doll','raincoat','ram_(animal)','raspberry','rat','razorblade','reamer_(juicer)','rearview_mirror',
        'receipt','recliner','record_player','reflector','remote_control','rhinoceros','rib_(food)','rifle','ring','river_boat','road_map','robe','rocking_chair','rodent',
        'roller_skate','Rollerblade','rolling_pin','root_beer','router_(computer_equipment)','rubber_band','runner_(carpet)','plastic_bag','saddle_(on_an_animal)','saddle_blanket',
        'saddlebag','safety_pin','sail','salad','salad_plate','salami','salmon_(fish)','salmon_(food)','salsa','saltshaker','sandal_(type_of_shoe)','sandwich','satchel','saucepan',
        'saucer','sausage','sawhorse','saxophone','scale_(measuring_instrument)','scarecrow','scarf','school_bus','scissors','scoreboard','scraper','screwdriver','scrubbing_brush',
        'sculpture','seabird','seahorse','seaplane','seashell','sewing_machine','shaker','shampoo','shark','sharpener','Sharpie','shaver_(electric)','shaving_cream','shawl',
        'shears','sheep','shepherd_dog','sherbert','shield','shirt','shoe','shopping_bag','shopping_cart','short_pants','shot_glass','shoulder_bag','shovel','shower_head',
        'shower_cap','shower_curtain','shredder_(for_paper)','signboard','silo','sink','skateboard','skewer','ski','ski_boot','ski_parka','ski_pole','skirt','skullcap','sled',
        'sleeping_bag','sling_(bandage)','slipper_(footwear)','smoothie','snake','snowboard','snowman','snowmobile','soap','soccer_ball','sock','sofa','softball','solar_array',
        'sombrero','soup','soup_bowl','soupspoon','sour_cream','soya_milk','space_shuttle','sparkler_(fireworks)','spatula','spear','spectacles','spice_rack','spider','crawfish',
        'sponge','spoon','sportswear','spotlight','squid_(food)','squirrel','stagecoach','stapler_(stapling_machine)','starfish','statue_(sculpture)','steak_(food)','steak_knife',
        'steering_wheel','stepladder','step_stool','stereo_(sound_system)','stew','stirrer','stirrup','stool','stop_sign','brake_light','stove','strainer','strap','straw_(for_drinking)',
        'strawberry','street_sign','streetlight','string_cheese','stylus','subwoofer','sugar_bowl','sugarcane_(plant)','suit_(clothing)','sunflower','sunglasses','sunhat','surfboard',
        'sushi','mop','sweat_pants','sweatband','sweater','sweatshirt','sweet_potato','swimsuit','sword','syringe','Tabasco_sauce','table-tennis_table','table','table_lamp','tablecloth',
        'tachometer','taco','tag','taillight','tambourine','army_tank','tank_(storage_vessel)','tank_top_(clothing)','tape_(sticky_cloth_or_paper)','tape_measure','tapestry','tarp','tartan',
        'tassel','tea_bag','teacup','teakettle','teapot','teddy_bear','telephone','telephone_booth','telephone_pole','telephoto_lens','television_camera','television_set',
        'tennis_ball','tennis_racket','tequila','thermometer','thermos_bottle','thermostat','thimble','thread','thumbtack','tiara','tiger','tights_(clothing)','timer',
        'tinfoil','tinsel','tissue_paper','toast_(food)','toaster','toaster_oven','toilet','toilet_tissue','tomato','tongs','toolbox','toothbrush','toothpaste','toothpick',
        'cover','tortilla','tow_truck','towel','towel_rack','toy','tractor_(farm_equipment)','traffic_light','dirt_bike','trailer_truck','train_(railroad_vehicle)',
        'trampoline','tray','trench_coat','triangle_(musical_instrument)','tricycle','tripod','trousers','truck','truffle_(chocolate)','trunk','vat','turban','turkey_(food)',
        'turnip','turtle','turtleneck_(clothing)','typewriter','umbrella','underwear','unicycle','urinal','urn','vacuum_cleaner','vase','vending_machine','vent','vest',
        'videotape','vinegar','violin','vodka','volleyball','vulture','waffle','waffle_iron','wagon','wagon_wheel','walking_stick','wall_clock','wall_socket','wallet','walrus','wardrobe','washbasin',
        'automatic_washer','watch','water_bottle','water_cooler','water_faucet','water_heater','water_jug','water_gun','water_scooter','water_ski','water_tower','watering_can','watermelon',
        'weathervane','webcam','wedding_cake','wedding_ring','wet_suit','wheel','wheelchair','whipped_cream','whistle','wig','wind_chime','windmill','window_box_(for_plants)',
        'windshield_wiper','windsock','wine_bottle','wine_bucket','wineglass','blinder_(for_horses)','wok','wolf','wooden_spoon','wreath','wrench','wristband','wristlet','yacht','yogurt','yoke_(animal_equipment)',
        'zebra','zucchini' 
    ]
lvis_common = [ 'aerosol_can', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'aquarium', 'armband',
        'artichoke', 'ashtray', 'asparagus', 'atomizer', 'award', 'basketball_backboard', 'bagel', 'bamboo', 'Band_Aid',
        'bandage', 'barrette', 'barrow', 'basketball', 'bat_(animal)', 'bathrobe', 'battery', 'bead', 'bean_curd', 'beanbag',
        'beer_can', 'beret', 'bib', 'binder', 'binoculars', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'black_sheep',
        'blackberry', 'blazer', 'bobbin', 'bobby_pin', 'boiled_egg', 'deadbolt', 'bookcase', 'booklet', 'bottle_opener', 'bouquet',
        'bowler_hat', 'suspenders', 'brassiere', 'bread-bin', 'briefcase', 'broom', 'brownie', 'brussels_sprouts', 'bull', 'bulldog',
        'bullet_train', 'bulletin_board', 'bullhorn', 'bunk_bed', 'business_card', 'butterfly', 'cabin_car', 'calculator', 'calf', 'camcorder',
        'camel', 'camera_lens', 'camper_(vehicle)', 'can_opener', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
        'cape', 'cappuccino', 'identity_card', 'card', 'cardigan', 'horse_carriage', 'cart', 'carton', 'cash_register', 'cast',
        'cayenne_(spice)', 'CD_player', 'cherry', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'crisp_(potato_chip)',
        'chocolate_bar', 'chocolate_cake', 'slide', 'cigarette_case', 'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clothes_hamper',
        'clothespin', 'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffeepot', 'coin', 'colander', 'coleslaw', 'pacifier',
        'corkscrew', 'cornet', 'cornice', 'corset', 'costume', 'cowbell', 'crab_(animal)', 'cracker', 'crayon', 'crescent_roll',
        'crib', 'crock_pot', 'crow', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crutch', 'cub_(animal)', 'cube',
        'cufflink', 'trophy_cup', 'dartboard', 'deer', 'dental_floss', 'diaper', 'dish_antenna', 'dishrag', 'dolphin', 'domestic_ass',
        'doormat', 'underdrawers', 'dress_hat', 'drill', 'drum_(musical_instrument)', 'duckling', 'duct_tape', 'dumpster',
        'eagle', 'easel', 'egg_yolk', 'eggbeater', 'eggplant', 'elk', 'envelope', 'eraser', 'Ferris_wheel', 'ferry', 'fighter_jet',
        'file_cabinet', 'fire_hose', 'fish_(food)', 'fishing_rod', 'flamingo', 'flannel', 'flap', 'flashlight', 'flipper_(footwear)',
        'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'footstool', 'forklift', 'freight_car',
        'French_toast', 'freshener', 'frog', 'fruit_juice', 'garbage_truck', 'garden_hose', 'gargle', 'garlic', 'gazelle', 'gelatin',
        'giant_panda', 'gift_wrap', 'ginger', 'cincture', 'globe', 'goat', 'golf_club', 'golfcart', 'goose', 'grater', 'gravestone',
        'grizzly', 'grocery_bag', 'gull', 'gun', 'hairnet', 'hairpin', 'hamburger', 'hammer', 'hammock', 'hamster', 'handcart',
        'handkerchief', 'veil', 'headscarf', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'highchair', 'hog', 'honey',
        'hot_sauce', 'hummingbird', 'icecream', 'ice_maker', 'igniter', 'iron_(for_clothing)', 'ironing_board', 'jam', 'jeep',
        'jet_plane', 'jewelry', 'jumpsuit', 'kayak', 'kettle', 'kilt', 'kimono', 'kitten', 'kiwi_fruit', 'ladle', 'ladybug', 'lantern',
        'legging_(clothing)', 'Lego', 'lion', 'lip_balm', 'lizard', 'lollipop', 'loveseat', 'mail_slot', 'mandarin_orange', 'manger',
        'mashed_potato', 'mat_(gym_equipment)', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'mitten',
        'mixer_(kitchen_tool)', 'money', 'monkey', 'muffin', 'musical_instrument', 'needle', 'nest', 'newsstand', 'nightshirt',
        'noseband_(for_animals)', 'notepad', 'oil_lamp', 'olive_oil', 'orange_juice', 'ostrich', 'overalls_(clothing)', 'owl',
        'packet', 'pad', 'padlock', 'paintbrush', 'palette', 'pancake', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
        'parka', 'parrot', 'passenger_car_(part_of_a_train)', 'passport', 'pea_(food)', 'peach', 'peanut_butter', 'peeler_(tool_for_fruit_and_vegetables)',
        'pelican', 'penguin', 'pepper_mill', 'perfume', 'pet', 'pew_(church_bench)', 'phonograph_record', 'pie', 'pigeon', 'pinecone',
        'pita_(bread)', 'platter', 'pliers', 'pocketknife', 'poker_(fire_stirring_tool)', 'pony', 'postbox_(public)', 'postcard', 'potholder',
        'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'projectile_(weapon)', 'projector', 'pumpkin', 'puppy', 'rabbit', 'racket',
        'radio_receiver', 'radish', 'raft', 'raincoat', 'ram_(animal)', 'raspberry', 'razorblade', 'reamer_(juicer)', 'receipt', 'recliner',
        'record_player', 'rhinoceros', 'rifle', 'robe', 'rocking_chair', 'rolling_pin', 'router_(computer_equipment)', 'runner_(carpet)',
        'saddlebag', 'salami', 'salmon_(fish)', 'salsa', 'school_bus', 'screwdriver', 'sculpture', 'seabird', 'seahorse', 'seashell',
        'sewing_machine', 'shaker', 'shampoo', 'shark', 'shaving_cream', 'shield', 'shopping_cart', 'shovel', 'silo', 'skewer', 'sled',
        'sleeping_bag', 'slipper_(footwear)', 'snowman', 'snowmobile', 'solar_array', 'soupspoon', 'sour_cream', 'spice_rack', 'spider',
        'sponge', 'sportswear', 'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish', 'steak_(food)', 'step_stool', 'stereo_(sound_system)',
        'strainer', 'sunflower', 'sunhat', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweet_potato', 'sword', 'table_lamp', 'tape_measure',
        'tapestry', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'telephone_booth', 'television_camera', 'thermometer', 'thermos_bottle',
        'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinsel', 'toast_(food)', 'toolbox', 'tortilla', 'tow_truck',
        'tractor_(farm_equipment)', 'dirt_bike', 'tricycle', 'trunk', 'turban', 'turkey_(food)', 'turtle', 'turtleneck_(clothing)', 'typewriter',
        'urn', 'vacuum_cleaner', 'vending_machine', 'videotape', 'volleyball', 'waffle', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock',
        'automatic_washer', 'water_cooler', 'water_faucet', 'water_jug', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'webcam',
        'wedding_cake', 'wedding_ring', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
        'windsock', 'wine_bucket', 'wok', 'wooden_spoon', 'wreath', 'wrench', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zucchini'
    ]
lvis_frequent = [ 'air_conditioner', 'airplane', 'alarm_clock', 'antenna', 'apple', 'apron', 'armchair', 'trash_can', 'avocado', 'awning',
        'baby_buggy', 'backpack', 'handbag', 'suitcase', 'ball', 'balloon', 'banana', 'bandanna', 'banner', 'barrel', 'baseball_base',
        'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'bath_mat', 'bath_towel', 'bathtub', 'beanie', 'bear',
        'bed', 'bedspread', 'cow', 'beef_(food)', 'beer_bottle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'bicycle', 'visor',
        'billboard', 'bird', 'birthday_cake', 'blackboard', 'blanket', 'blender', 'blinker', 'blouse', 'blueberry', 'boat', 'bolt', 'book',
        'boot', 'bottle', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'box', 'bracelet', 'bread', 'bridal_gown', 'broccoli', 'bucket', 'bun',
        'buoy', 'bus_(vehicle)', 'butter', 'button', 'cab_(taxi)', 'cabinet', 'cake', 'calendar', 'camera', 'can', 'candle', 'candle_holder',
        'cap_(headwear)', 'bottle_cap', 'car_(automobile)', 'railcar_(part_of_a_train)', 'carrot', 'tote_bag', 'cat', 'cauliflower', 'celery',
        'cellular_telephone', 'chair', 'chandelier', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'cigarette', 'cistern', 'clock',
        'clock_tower', 'coaster', 'coat', 'coffee_maker', 'coffee_table', 'computer_keyboard', 'condiment', 'cone', 'control', 'cookie',
        'cooler_(for_food)', 'cork_(bottle_plug)', 'edible_corn', 'cowboy_hat', 'crate', 'crossbar', 'crumb', 'cucumber', 'cup', 'cupboard',
        'cupcake', 'curtain', 'cushion', 'deck_chair', 'desk', 'dining_table', 'dish', 'dishtowel', 'dishwasher', 'dispenser', 'Dixie_cup', 'dog',
        'dog_collar', 'doll', 'doorknob', 'doughnut', 'drawer', 'dress', 'dress_suit', 'dresser', 'duck', 'duffel_bag', 'earphone', 'earring', 'egg',
        'refrigerator', 'elephant', 'fan', 'faucet', 'figurine', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fireplace', 'fireplug', 'fish',
        'flag', 'flagpole', 'flip-flop_(sandal)', 'flower_arrangement', 'fork', 'frisbee', 'frying_pan', 'giraffe', 'glass_(drink_container)', 'glove',
        'goggles', 'grape', 'green_bean', 'green_onion', 'grill', 'guitar', 'hairbrush', 'ham', 'hair_dryer', 'hand_towel', 'handle', 'hat', 'headband',
        'headboard', 'headlight', 'helmet', 'hinge', 'home_plate_(baseball)', 'fume_hood', 'hook', 'horse', 'hose', 'polar_bear', 'iPod', 'jacket',
        'jar', 'jean', 'jersey', 'key', 'kitchen_sink', 'kite', 'knee_pad', 'knife', 'knob', 'ladder', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade',
        'lanyard', 'laptop_computer', 'latch', 'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lime', 'log',
        'speaker_(stero_equipment)', 'magazine', 'magnet', 'mailbox_(at_home)', 'manhole', 'map', 'marker', 'mask', 'mast', 'mattress', 'microphone',
        'microwave_oven', 'milk', 'minivan', 'mirror', 'monitor_(computer_equipment) computer_monitor', 'motor', 'motor_scooter', 'motorcycle',
        'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'mug', 'mushroom', 'napkin', 'necklace', 'necktie', 'newspaper', 'notebook',
        'nut', 'oar', 'onion', 'orange_(fruit)', 'ottoman', 'oven', 'paddle', 'painting', 'pajamas', 'pan_(for_cooking)', 'paper_plate', 'paper_towel',
        'parking_meter', 'pastry', 'pear', 'pen', 'pencil', 'pepper', 'person', 'piano', 'pickle', 'pickup_truck', 'pillow', 'pineapple', 'pipe',
        'pitcher_(vessel_for_liquid)', 'pizza', 'place_mat', 'plate', 'pole', 'polo_shirt', 'pop_(soda)', 'poster', 'pot', 'flowerpot', 'potato',
        'printer', 'propeller', 'quilt', 'radiator', 'rearview_mirror', 'reflector', 'remote_control', 'ring', 'rubber_band', 'plastic_bag',
        'saddle_(on_an_animal)', 'saddle_blanket', 'sail', 'salad', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage',
        'scale_(measuring_instrument)', 'scarf', 'scissors', 'scoreboard', 'scrubbing_brush', 'sheep', 'shirt', 'shoe', 'shopping_bag', 'short_pants',
        'shoulder_bag', 'shower_head', 'shower_curtain', 'signboard', 'sink', 'skateboard', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt',
        'snowboard', 'soap', 'soccer_ball', 'sock', 'sofa', 'soup', 'spatula', 'spectacles', 'spoon', 'statue_(sculpture)', 'steering_wheel', 'stirrup',
        'stool', 'stop_sign', 'brake_light', 'stove', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)',
        'sunglasses', 'surfboard', 'sweater', 'sweatshirt', 'swimsuit', 'table', 'tablecloth', 'tag', 'taillight', 'tank_(storage_vessel)',
        'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tarp', 'teapot', 'teddy_bear', 'telephone', 'telephone_pole', 'television_set',
        'tennis_ball', 'tennis_racket', 'thermostat', 'tinfoil', 'tissue_paper', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato',
        'tongs', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'towel', 'towel_rack', 'toy', 'traffic_light', 'trailer_truck',
        'train_(railroad_vehicle)', 'tray', 'tripod', 'trousers', 'truck', 'umbrella', 'underwear', 'urinal', 'vase', 'vent', 'vest', 'wall_socket',
        'wallet', 'watch', 'water_bottle', 'watermelon', 'weathervane', 'wet_suit', 'wheel', 'windshield_wiper', 'wine_bottle', 'wineglass',
        'blinder_(for_horses)', 'wristband', 'wristlet', 'zebra'
    ]
lvis_rare = [ 'applesauce', 'apricot', 'arctic_(type_of_shoe)', 'armoire', 'armor', 'ax', 'baboon', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'banjo',
        'barbell', 'barge', 'bass_horn', 'batter_(food)', 'beachball', 'bedpan', 'beeper', 'beetle', 'Bible', 'birthday_card', 'pirate_flag', 'blimp',
        'gameboard', 'bob', 'bolo_tie', 'bonnet', 'bookmark', 'boom_microphone', 'bow_(weapon)', 'pipe_bowl', 'bowling_ball', 'boxing_glove', 'brass_plaque',
        'breechcloth', 'broach', 'bubble_gum', 'horse_buggy', 'bulldozer', 'bulletproof_vest', 'burrito', 'cabana', 'locker', 'candy_bar', 'canteen',
        'elevator_car', 'car_battery', 'cargo_ship', 'carnation', 'casserole', 'cassette', 'chain_mail', 'chaise_longue', 'chalice', 'chap', 'checkbook',
        'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker_chip', 'chocolate_milk', 'chocolate_mousse', 'cider', 'cigar_box', 'clarinet',
        'cleat_(for_securing_rope)', 'clementine', 'clippers_(for_plants)', 'cloak', 'clutch_bag', 'cockroach', 'cocoa_(beverage)', 'coil', 'coloring_material',
        'combination_lock', 'comic_book', 'compass', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cooking_utensil', 'corkboard', 'cornbread',
        'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream_pitcher', 'crouton', 'crowbar', 'hair_curler', 'curling_iron', 'cylinder', 'cymbal',
        'dagger', 'dalmatian', 'date_(fruit)', 'detergent', 'diary', 'die', 'dinghy', 'tux', 'dishwasher_detergent', 'diving_board', 'dollar', 'dollhouse',
        'dove', 'dragonfly', 'drone', 'dropper', 'drumstick', 'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg_roll', 'electric_chair', 'escargot',
        'eyepatch', 'falcon', 'fedora', 'ferret', 'fig_(fruit)', 'file_(tool)', 'first-aid_kit', 'fishbowl', 'flash', 'fleece', 'football_helmet', 'fudge',
        'funnel', 'futon', 'gag', 'garbage', 'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish', 'gondola_(boat)', 'gorilla', 'gourd', 'gravy_boat',
        'griddle', 'grits', 'halter_top', 'hamper', 'hand_glass', 'handcuff', 'handsaw', 'hardback_book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus',
        'hockey_stick', 'hookah', 'hornet', 'hot-air_balloon', 'hotplate', 'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice_pack', 'ice_skate', 'inhaler',
        'jelly_bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard', 'kitchen_table', 'knitting_needle', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'lamb-chop',
        'lasagna', 'lawn_mower', 'leather', 'legume', 'lemonade', 'lightning_rod', 'limousine', 'liquor', 'machine_gun', 'mallard', 'mallet', 'mammoth', 'manatee',
        'martini', 'mascot', 'masher', 'matchbox', 'microscope', 'milestone', 'milk_can', 'milkshake', 'mint_candy', 'motor_vehicle', 'music_stool', 'nailfile',
        'neckerchief', 'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)', 'octopus_(animal)', 'omelet', 'inkpad', 'pan_(metal_container)', 'pantyhose',
        'papaya', 'paperback_book', 'paperweight', 'parchment', 'passenger_ship', 'patty_(food)', 'wooden_leg', 'pegboard', 'pencil_box', 'pencil_sharpener',
        'pendulum', 'pennant', 'penny_(coin)', 'persimmon', 'phonebook', 'piggy_bank', 'pin_(non_jewelry)', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pistol',
        'pitchfork', 'playpen', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'poncho', 'pool_table', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog',
        'puncher', 'puppet', 'quesadilla', 'quiche', 'race_car', 'radar', 'rag_doll', 'rat', 'rib_(food)', 'river_boat', 'road_map', 'rodent', 'roller_skate',
        'Rollerblade', 'root_beer', 'safety_pin', 'salad_plate', 'salmon_(food)', 'satchel', 'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper',
        'seaplane', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shawl', 'shears', 'shepherd_dog', 'sherbert', 'shot_glass', 'shower_cap', 'shredder_(for_paper)',
        'skullcap', 'sling_(bandage)', 'smoothie', 'snake', 'softball', 'sombrero', 'soup_bowl', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spear',
        'crawfish', 'squid_(food)', 'stagecoach', 'steak_knife', 'stepladder', 'stew', 'stirrer', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl',
        'sugarcane_(plant)', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'tachometer', 'taco', 'tambourine', 'army_tank', 'telephoto_lens', 'tequila',
        'thimble', 'trampoline', 'trench_coat', 'triangle_(musical_instrument)', 'truffle_(chocolate)', 'vat', 'turnip', 'unicycle', 'vinegar', 'violin', 'vodka',
        'vulture', 'waffle_iron', 'walrus', 'wardrobe', 'washbasin', 'water_heater', 'water_gun', 'wolf'
    ]


METAINFO = {
    'lvis': tuple(lvis),
    'lvis_common': tuple(lvis_common),
    'lvis_frequent': tuple(lvis_frequent),
    'lvis_rare': tuple(lvis_rare),
    'lvis_minival': tuple(lvis),
    'lvis_minival_common': tuple(lvis_common),
    'lvis_minival_frequent': tuple(lvis_frequent),
    'lvis_minival_rare': tuple(lvis_rare),
    'default_classes':
        (
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
             'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
             'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
             'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
             'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
             'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
             'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
             'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
             'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ),
    'pascal_voc_split_1': list(set(voc_all_classes_1) - set(voc_split_1_seen_classes)), # unseen classes split 1
    'pascal_voc_split_2': list(set(voc_all_classes_2) - set(voc_split_2_seen_classes)), # unseen classes split 2
    'pascal_voc_split_3': list(set(voc_all_classes_3) - set(voc_split_3_seen_classes)), # unseen classes split 3
    'few_shot_classes':
        (
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv'
        ),
    'coco_semantic_split_1':
        (
            'person', 'airplane', 'boat', 'parking meter', 'dog', 'elephant', 'backpack',
            'suitcase', 'sports ball', 'skateboard', 'wine glass', 'spoon', 'sandwich',
            'hot dog', 'chair', 'dining table', 'mouse', 'microwave', 'refrigerator',
            'scissors'
        ),
    'coco_semantic_split_2':
        (
            'bicycle', 'bus', 'traffic light', 'bench', 'horse', 'bear', 'umbrella',
            'frisbee', 'kite', 'surfboard', 'cup', 'bowl', 'orange', 'pizza', 'couch',
            'toilet', 'remote', 'oven', 'book', 'teddy bear'
        ),
    'coco_semantic_split_3':
        (
            'car', 'train', 'fire hydrant', 'bird', 'sheep', 'zebra', 'handbag',
            'skis', 'baseball bat', 'tennis racket', 'fork', 'banana', 'broccoli',
            'donut', 'potted plant', 'tv', 'keyboard', 'toaster', 'clock', 'hair drier'
        ),
    'coco_semantic_split_4':
        (
            'motorcycle', 'truck', 'stop sign', 'cat', 'cow', 'giraffe', 'tie',
            'snowboard', 'baseball glove', 'bottle', 'knife', 'apple', 'carrot',
            'cake', 'bed', 'laptop', 'cell phone', 'sink', 'vase', 'toothbrush'
        ),
}