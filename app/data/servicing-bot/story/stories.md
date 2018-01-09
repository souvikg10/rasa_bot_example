## path greet
* _greet
  - utter_greet

## path card lost for credit
* _greet              
  - utter_greet
* _card_lost               
  - utter_card_lost_type
* _card_lost[card_type="credit"] 
  - utter_card_lost 
* _goodbye              
  - utter_goodbye

## path card lost for debit             
* _greet              
  - utter_greet
* _card_lost               
  - utter_card_lost_type
* _card_lost[card_type="debit"] 
  - utter_card_lost 
* _goodbye              
  - utter_goodbye  

## path credit              
* _greet              
  - utter_greet
* _card_lost[card_type="credit"]               
  - utter_card_lost
* _goodbye              
  - utter_goodbye


## path debit               
* _greet              
  - utter_greet
* _card_lost[card_type="debit"]               
  - utter_card_lost
* _goodbye              
  - utter_goodbye

## say goodbye
* _goodbye
  - utter_goodbye
