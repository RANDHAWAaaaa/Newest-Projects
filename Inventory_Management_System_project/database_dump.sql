-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema inventory_system
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema inventory_system
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `inventory_system` DEFAULT CHARACTER SET utf8 ;
USE `inventory_system` ;

-- -----------------------------------------------------
-- Table `inventory_system`.`employee`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`employee` (
  `employee_id` INT(11) NOT NULL AUTO_INCREMENT,
  `emp_name` VARCHAR(45) NULL DEFAULT NULL,
  `emp_address` VARCHAR(45) NULL DEFAULT NULL,
  `emp_telephone` VARCHAR(45) NULL DEFAULT NULL,
  PRIMARY KEY (`employee_id`))
ENGINE = InnoDB
AUTO_INCREMENT = 11
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `inventory_system`.`supplier_details`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`supplier_details` (
  `supplier_id` INT(11) NOT NULL AUTO_INCREMENT,
  `supplier_name` VARCHAR(45) NULL DEFAULT NULL,
  PRIMARY KEY (`supplier_id`))
ENGINE = InnoDB
AUTO_INCREMENT = 6
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `inventory_system`.`order`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`order` (
  `order_id` INT(11) NOT NULL AUTO_INCREMENT,
  `orderdate` DATE NULL DEFAULT NULL,
  `employee_employee_id` INT(11) NOT NULL,
  `supplier_id` INT(11) NOT NULL,
  PRIMARY KEY (`order_id`),
  INDEX `FK_orders_emp_id_idx` (`employee_employee_id` ASC),
  INDEX `FK_orders_supplier_id_idx` (`supplier_id` ASC),
  CONSTRAINT `FK_orders_emp_id`
    FOREIGN KEY (`employee_employee_id`)
    REFERENCES `inventory_system`.`employee` (`employee_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `FK_orders_supplier_id`
    FOREIGN KEY (`supplier_id`)
    REFERENCES `inventory_system`.`supplier_details` (`supplier_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
AUTO_INCREMENT = 6
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `inventory_system`.`product`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`product` (
  `p_id` INT(11) NOT NULL AUTO_INCREMENT,
  `p_name` VARCHAR(45) NOT NULL,
  `quantity` INT(11) NULL DEFAULT '0',
  PRIMARY KEY (`p_id`))
ENGINE = InnoDB
AUTO_INCREMENT = 12
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `inventory_system`.`purchase`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`purchase` (
  `purchase_id` INT(11) NOT NULL AUTO_INCREMENT,
  `p_id` INT(11) NOT NULL,
  `price` DOUBLE NOT NULL,
  `pur_date` DATE NOT NULL,
  `quantity` INT(11) NULL DEFAULT NULL,
  PRIMARY KEY (`purchase_id`),
  INDEX `FK_purchase_p_id_idx` (`p_id` ASC),
  CONSTRAINT `FK_purchase_p_id`
    FOREIGN KEY (`p_id`)
    REFERENCES `inventory_system`.`product` (`p_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
AUTO_INCREMENT = 6
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `inventory_system`.`sales`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`sales` (
  `s_id` INT(11) NOT NULL AUTO_INCREMENT,
  `price` DOUBLE NOT NULL,
  `sdate` DATE NULL DEFAULT NULL,
  `quantity` INT(11) NULL DEFAULT '0',
  `employee_employee_id` INT(11) NOT NULL,
  `supplier_details_supplier_id` INT(11) NOT NULL,
  `product_p_id` INT(11) NOT NULL,
  PRIMARY KEY (`s_id`),
  INDEX `fk_sales_supplier_details1_idx` (`supplier_details_supplier_id` ASC),
  INDEX `FK_sales_p_id_idx` (`product_p_id` ASC),
  INDEX `FK_sales_emp_id_idx` (`employee_employee_id` ASC),
  CONSTRAINT `FK_sales_emp_id`
    FOREIGN KEY (`employee_employee_id`)
    REFERENCES `inventory_system`.`employee` (`employee_id`)
    ON DELETE NO ACTION
    ON UPDATE CASCADE,
  CONSTRAINT `FK_sales_p_id`
    FOREIGN KEY (`product_p_id`)
    REFERENCES `inventory_system`.`product` (`p_id`)
    ON DELETE NO ACTION
    ON UPDATE CASCADE,
  CONSTRAINT `fk_sales_supplier_details1`
    FOREIGN KEY (`supplier_details_supplier_id`)
    REFERENCES `inventory_system`.`supplier_details` (`supplier_id`)
    ON DELETE NO ACTION
    ON UPDATE CASCADE)
ENGINE = InnoDB
AUTO_INCREMENT = 7
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `inventory_system`.`supplier_invoice`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`supplier_invoice` (
  `s_v_id` INT(11) NOT NULL AUTO_INCREMENT,
  `invoice_date` DATE NULL DEFAULT NULL,
  `Total_price` DOUBLE NULL DEFAULT NULL,
  `Amount_paid` DOUBLE NULL DEFAULT NULL,
  `Pending_amount` DOUBLE NULL DEFAULT NULL,
  `Status` VARCHAR(45) NULL DEFAULT NULL,
  `supplier_details_supplier_id` INT(11) NOT NULL,
  `product_p_id` INT(11) NOT NULL,
  PRIMARY KEY (`s_v_id`),
  INDEX `fk_supplier_invoice_supplier_details1_idx` (`supplier_details_supplier_id` ASC),
  INDEX `FK_supplier_invoice_p_id_idx` (`product_p_id` ASC),
  CONSTRAINT `FK_supplier_invoice_p_id`
    FOREIGN KEY (`product_p_id`)
    REFERENCES `inventory_system`.`product` (`p_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_supplier_invoice_supplier_details1`
    FOREIGN KEY (`supplier_details_supplier_id`)
    REFERENCES `inventory_system`.`supplier_details` (`supplier_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
AUTO_INCREMENT = 6
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `inventory_system`.`supplier_payment`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`supplier_payment` (
  `suppstat_id` INT(11) NOT NULL AUTO_INCREMENT,
  `supplier_id` INT(11) NULL DEFAULT NULL,
  `p_id` INT(11) NULL DEFAULT NULL,
  `quantity` INT(11) NULL DEFAULT NULL,
  `Total_price` DOUBLE NULL DEFAULT '0',
  PRIMARY KEY (`suppstat_id`),
  INDEX `FK_supplier_status_supplier_id_idx` (`supplier_id` ASC),
  INDEX `FK_supplier_ststus_p_id_idx` (`p_id` ASC),
  CONSTRAINT `FK_supplier_status_supplier_id`
    FOREIGN KEY (`supplier_id`)
    REFERENCES `inventory_system`.`supplier_details` (`supplier_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `FK_supplier_ststus_p_id`
    FOREIGN KEY (`p_id`)
    REFERENCES `inventory_system`.`product` (`p_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
AUTO_INCREMENT = 6
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `inventory_system`.`supplier_request_product`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`supplier_request_product` (
  `supplier_id` INT(11) NOT NULL,
  `p_id` INT(11) NULL DEFAULT NULL,
  `supplier_sales_id` INT(11) NOT NULL AUTO_INCREMENT,
  `supplierqty` INT(11) NULL DEFAULT NULL,
  PRIMARY KEY (`supplier_sales_id`),
  INDEX `FK_supplier_request_p_id_idx` (`p_id` ASC),
  INDEX `FK_supplier_request_supplier_id_idx` (`supplier_id` ASC),
  CONSTRAINT `FK_supplier_request_p_id`
    FOREIGN KEY (`p_id`)
    REFERENCES `inventory_system`.`product` (`p_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `FK_supplier_request_supplier_id`
    FOREIGN KEY (`supplier_id`)
    REFERENCES `inventory_system`.`supplier_details` (`supplier_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
AUTO_INCREMENT = 8
DEFAULT CHARACTER SET = utf8;

USE `inventory_system` ;

-- -----------------------------------------------------
-- Placeholder table for view `inventory_system`.`highest_sold_product`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`highest_sold_product` (`No_of_times_sold` INT, `p_name` INT);

-- -----------------------------------------------------
-- Placeholder table for view `inventory_system`.`outstanding _amount`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `inventory_system`.`outstanding _amount` (`supplier_name` INT, `p_name` INT, `Total_price` INT, `Pending_amount` INT);

-- -----------------------------------------------------
-- procedure Products_brought_by_supplier
-- -----------------------------------------------------

DELIMITER $$
USE `inventory_system`$$
CREATE DEFINER=`root`@`localhost` PROCEDURE `Products_brought_by_supplier`(IN supplierid int)
BEGIN
select si.invoice_date,sd.supplier_id,sd.supplier_name,p.p_name,si.`status` from supplier_details sd 
inner join supplier_invoice si
on si.supplier_details_supplier_id = sd.supplier_id
inner join product p
on p.p_id = si.product_p_id
where sd.supplier_id = supplierid;
END$$

DELIMITER ;

-- -----------------------------------------------------
-- procedure Update_Pending_amount
-- -----------------------------------------------------

DELIMITER $$
USE `inventory_system`$$
CREATE DEFINER=`root`@`localhost` PROCEDURE `Update_Pending_amount`(IN invoiceid int,IN Price int)
BEGIN
update supplier_invoice
set Amount_paid = Amount_paid + Price
where s_v_id = invoiceid;
END$$

DELIMITER ;

-- -----------------------------------------------------
-- function calculate_loss
-- -----------------------------------------------------

DELIMITER $$
USE `inventory_system`$$
CREATE DEFINER=`root`@`localhost` FUNCTION `calculate_loss`(id int) RETURNS varchar(11) CHARSET utf8
BEGIN
DECLARE pur varchar(10);
DECLARE sale varchar(10);
DECLARE profit integer(10);
DECLARE loss varchar(10);
DECLARE `status` varchar(10);
set pur = (select  SUM(price* quantity) from purchase where p_id = id group by p_id);
set sale = (select  SUM(price* quantity) from sales where product_p_id = id group by product_p_id);
if pur > sale
then 
set loss = pur - sale;
else
set loss = 'no loss';
end if;
Return loss;
END$$

DELIMITER ;

-- -----------------------------------------------------
-- function calculate_profit
-- -----------------------------------------------------

DELIMITER $$
USE `inventory_system`$$
CREATE DEFINER=`root`@`localhost` FUNCTION `calculate_profit`(id int) RETURNS varchar(11) CHARSET utf8
BEGIN
DECLARE pur varchar(10);
DECLARE sale varchar(10);
DECLARE profit varchar(10);
DECLARE loss integer(10);
DECLARE `status` varchar(10);

set pur = (select  SUM(price* quantity) from purchase where p_id = id group by p_id);
set sale = (select  SUM(price* quantity) from sales where product_p_id = id group by product_p_id);
if sale > pur 
then 
set profit = (sale - pur); 
else 
set profit = 'no profit';
end if;
Return profit;
END$$

DELIMITER ;

-- -----------------------------------------------------
-- procedure save_product
-- -----------------------------------------------------

DELIMITER $$
USE `inventory_system`$$
CREATE DEFINER=`root`@`localhost` PROCEDURE `save_product`(productname varchar(50))
BEGIN
insert into product (pname) values(productname);
END$$

DELIMITER ;

-- -----------------------------------------------------
-- procedure save_purchase
-- -----------------------------------------------------

DELIMITER $$
USE `inventory_system`$$
CREATE DEFINER=`root`@`localhost` PROCEDURE `save_purchase`(IN id int,in price double,in dt date,in qty int)
BEGIN
insert into purchase (p_id, price, pur_date, quantity) 
values (id,price,dt,qty);
END$$

DELIMITER ;

-- -----------------------------------------------------
-- procedure save_sale
-- -----------------------------------------------------

DELIMITER $$
USE `inventory_system`$$
CREATE DEFINER=`root`@`localhost` PROCEDURE `save_sale`(IN id int,in price double,in dt date,in qty int)
BEGIN
insert into sales (p_id, price, sdate, quantity) 
values (id,price,dt,qty);
END$$

DELIMITER ;

-- -----------------------------------------------------
-- function supplier_product_amount
-- -----------------------------------------------------

DELIMITER $$
USE `inventory_system`$$
CREATE DEFINER=`root`@`localhost` FUNCTION `supplier_product_amount`(supplierid int) RETURNS varchar(20) CHARSET utf8
BEGIN
declare total int(11);
RETURN (select SUM(Total_price) from supplier_payment where supplier_id = supplierid group by supplier_id); 

END$$

DELIMITER ;

-- -----------------------------------------------------
-- View `inventory_system`.`highest_sold_product`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `inventory_system`.`highest_sold_product`;
USE `inventory_system`;
CREATE  OR REPLACE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` 
SQL SECURITY DEFINER VIEW `inventory_system`.`highest_sold_product` AS 
select count(`sp`.`p_id`) AS `No_of_times_sold`,`p`.`p_name` AS 
`p_name` from (`inventory_system`.`supplier_payment` `sp` join `inventory_system`.`product` `p` 
on((`sp`.`p_id` = `p`.`p_id`))) group by `p`.`p_id` order by count(`sp`.`p_id`) desc limit 1;

-- -----------------------------------------------------
-- View `inventory_system`.`outstanding _amount`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `inventory_system`.`outstanding _amount`;
USE `inventory_system`;
CREATE  OR REPLACE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` 
SQL SECURITY DEFINER VIEW `inventory_system`.`outstanding _amount` AS 
select `sd`.`supplier_name` AS `supplier_name`,`p`.`p_name` AS
 `p_name`,`si`.`Total_price` AS `Total_price`,`si`.`Pending_amount` AS `Pending_amount` 
 from ((`inventory_system`.`supplier_details` `sd` join `inventory_system`.`supplier_invoice` `si` 
 on((`sd`.`supplier_id` = `si`.`supplier_details_supplier_id`))) 
 join `inventory_system`.`product` `p` on((`si`.`product_p_id` = `p`.`p_id`))) 
 where (`si`.`Status` = 'Pending');
USE `inventory_system`;

DELIMITER $$
USE `inventory_system`$$
CREATE
DEFINER=`root`@`localhost`
TRIGGER `inventory_system`.`purchase_BEFORE_INSERT`
BEFORE INSERT ON `inventory_system`.`purchase`
FOR EACH ROW
begin
update product
set quantity = quantity + new.quantity
where p_id = new.p_id;

END$$

USE `inventory_system`$$
CREATE
DEFINER=`root`@`localhost`
TRIGGER `inventory_system`.`sales_AFTER_INSERT`
AFTER INSERT ON `inventory_system`.`sales`
FOR EACH ROW
BEGIN
insert into inventory_system.order
set orderdate = new.sdate,
employee_employee_id = new.employee_employee_id,
supplier_id = new.supplier_details_supplier_id;

insert into supplier_payment
set p_id = new.product_p_id ,
supplier_id = new.supplier_details_supplier_id,
quantity = new.quantity,
Total_price = new.quantity * new.price;
END$$

USE `inventory_system`$$
CREATE
DEFINER=`root`@`localhost`
TRIGGER `inventory_system`.`sales_BEFORE_INSERT`
BEFORE INSERT ON `inventory_system`.`sales`
FOR EACH ROW
BEGIN


set new.quantity = (select supplierqty from supplier_request_product s
where new.supplier_details_supplier_id = s.supplier_id and new.product_p_id = s.p_id );

update product
set quantity = quantity - new.quantity
where p_id = new.product_p_id and quantity > 0;
END$$

USE `inventory_system`$$
CREATE
DEFINER=`root`@`localhost`
TRIGGER `inventory_system`.`supplier_invoice_BEFORE_INSERT`
BEFORE INSERT ON `inventory_system`.`supplier_invoice`
FOR EACH ROW
BEGIN
set new.Total_price = (select Total_price from supplier_payment where supplier_id = new.supplier_details_supplier_id and p_id = new.product_p_id);
set new.Pending_amount = new.Total_price - new.Amount_paid;
if (new.Pending_amount > 0)
then
set new.Status = 'pending';
elseif(new.Pending_amount = 0)
then
set new.Status = 'Complete';
else
set new.Pending_amount = 0,new.Status = 'Complete',new.Amount_paid = new.Total_price;
end if ;
END$$

USE `inventory_system`$$
CREATE
DEFINER=`root`@`localhost`
TRIGGER `inventory_system`.`supplier_invoice_BEFORE_UPDATE`
BEFORE UPDATE ON `inventory_system`.`supplier_invoice`
FOR EACH ROW
BEGIN
set new.Total_price = (select Total_price from supplier_payment 
where supplier_id = new.supplier_details_supplier_id and p_id = new.product_p_id);
set new.Pending_amount = new.Total_price - new.Amount_paid;
if (new.Pending_amount > 0)
then
set new.Status = 'pending';
elseif(new.Pending_amount = 0)
then
set new.Status = 'Complete';
else
set new.Pending_amount = 0,new.Status = 'Complete',new.Amount_paid = new.Total_price;
end if ;
END$$
USE `inventory_system`$$
CREATE
DEFINER=`root`@`localhost`
TRIGGER `inventory_system`.`supplier_request_product_AFTER_INSERT`
AFTER INSERT ON `inventory_system`.`supplier_request_product`
FOR EACH ROW
BEGIN
update sales
set quantity = new.supplierqty
where supplier_details_supplier_id = new.supplier_id and product_p_id = new.p_id ; 
END$$


DELIMITER ;

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

use inventory_system;
-- inserting product details in the manufacturing inventory
insert into product (p_id,p_name) values (1,'Lenovo C260');
insert into product (p_id,p_name) values (2,'Sony VAIO SVF1521ASNB 15.5');
insert into product (p_id,p_name) values (3,'HP 14-Q001TU 14');
insert into product (p_id,p_name) values (4,'Samsung XE303C12-A01IN 11.6');
insert into product (p_id,p_name) values (5,'Samsung 43F4100 Plasma TV');

select * from product;


delete from employee where employee_id = 10;
-- inserting employee details of manufacturing company

insert into employee (employee_id,emp_name,emp_telephone,emp_address) values (1,'A','12345','peterborough street');
insert into employee (employee_id,emp_name,emp_telephone,emp_address) values (2,'B','12345','peterborough street');
insert into employee (employee_id,emp_name,emp_telephone,emp_address) values (3,'C','12345','peterborough street');
insert into employee (employee_id,emp_name,emp_telephone,emp_address) values (4,'D','12345','peterborough street');
insert into employee (employee_id,emp_name,emp_telephone,emp_address) values (5,'E','12345','peterborough street');

select * from employee;

-- insering supplier_details which are the customers of manufacturing company
insert into supplier_details values(1,'Birwa');
insert into supplier_details values(2,'Milony');
insert into supplier_details values(3,'Akshay');
insert into supplier_details values(4,'Ram');
insert into supplier_details values(5,'Mohan');

select * from supplier_details;

 -- inserting details to purchase particular product
insert into purchase(p_id, price, pur_date, quantity) 
values(1, 300, '2017-12-13', 100);
insert into purchase(p_id, price, pur_date, quantity) 
values (2, 2, 400, '2016-06-12', 100);
insert into purchase(p_id, price, pur_date, quantity) 
values (3, 400, '2017-06-12', 100);
insert into purchase(p_id, price, pur_date, quantity) 
values  (4, 600, '2017-08-30', 100);
insert into purchase(p_id, price, pur_date, quantity) 
values (5,200,'2017-10-23',100);

select * from purchase;

-- supplier requests the product which he wants to buy and mentions the quantity 
insert into supplier_request_product(supplier_id,p_id,supplierqty) values (1,2,7);
insert into supplier_request_product(supplier_id,p_id,supplierqty) values (1,3,7);
insert into supplier_request_product(supplier_id,p_id,supplierqty) values (1,1,8);
insert into supplier_request_product(supplier_id,p_id,supplierqty) values (2,3,10);
insert into supplier_request_product(supplier_id,p_id,supplierqty) values (2,2,10);
insert into supplier_request_product(supplier_id,p_id,supplierqty) values (3,4,9);
insert into supplier_request_product(supplier_id,p_id,supplierqty) values (4,5,5);
insert into supplier_request_product(supplier_id,p_id,supplierqty) values (5,3,101);
delete from supplier_request_product where supplier_sales_id =10;


select * from supplier_request_product;

-- if quantity mentioned by supplier is present in product_inventory then it is processed in sales
insert into sales(product_p_id, price, sdate, quantity,employee_employee_id,supplier_details_supplier_id) 
values (2,300,now(),3,3,1);
insert into sales(product_p_id, price, sdate, quantity,employee_employee_id,supplier_details_supplier_id) 
values (3,900,now(),2,2,2);
insert into sales(product_p_id, price, sdate, quantity,employee_employee_id,supplier_details_supplier_id) 
values (2,800,now(),2,1,2);
insert into sales(product_p_id, price, sdate, quantity,employee_employee_id,supplier_details_supplier_id) 
values (4,700,now(),2,1,3);
insert into sales(product_p_id, price, sdate, quantity,employee_employee_id,supplier_details_supplier_id) 
values (5,100,now(),2,4,4);
insert into sales(product_p_id, price, sdate, quantity,employee_employee_id,supplier_details_supplier_id) 
values (3,900,now(),2,4,5);
select * from sales;




-- once the sales details are inserted in the sales table then order table is updated for that particular supplier 
select * from `order`;

 -- once the order is placed then total price for that supplier is calculated 
select * from supplier_payment;

-- inserting amount paid for that supplier which calculates the pending_amount
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values('2017-12-13',900,1,1);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values('2017-12-17',800,1,2);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values('2017-08-13',900,2,3);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values('2017-10-30',1600,2,2);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values(now(),6000,3,4);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values(now(),100,4,5);

 
select * from supplier_invoice;

