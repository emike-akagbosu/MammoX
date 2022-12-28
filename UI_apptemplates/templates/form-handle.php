<?php

$name = $_POST['name'];
$visitor_email = $_POST['email'];
$message = $_POST['message'];

$email_from = 'fl1520@ic.ac.uk';
$email_subject = 'New MammoX Message';
$email_body = "User Name: $name.\n "
                 "User Email: $visitor_email.\n "
                 "Message: $message.\n";

$to = 'fl1520@ic.ac.uk';
$headers = "From: $email_from \r\n"
$headers .= "Reply-To: $visitor_emaIL\r\n";

mail($to,$email_subject,$email_body,$headers);

header("Location:contact.html")
?>