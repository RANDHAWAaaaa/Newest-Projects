use inventory_system;

-- inserting product details in the manufacturing inventory
 -- -----------------------------------------------------
-- 1. Product
-- -----------------------------------------------------
select * from product;


-- inserting employee details of manufacturing company
 -- -----------------------------------------------------
-- 2. Employee
-- -----------------------------------------------------
select * from employee;

-- insering supplier_details which are the customers of manufacturing company

 -- -----------------------------------------------------
-- 3. Supplier_details
-- -----------------------------------------------------
select * from supplier_details;


-- -----------------------------------------------------
-- 4. Purchase table with cost price of the product 
-- -----------------------------------------------------
select * from purchase;
call purchase_BEFORE_INSERT(); -- to increase the quantity in product table

-- supplier requests the product which he wants to buy and mentions the quantity
-- -----------------------------------------------------
-- 5. Supplier_request_product to order the product
-- -----------------------------------------------------
call supplier_request_product_AFTER_INSERT();-- assigns the quantity mentioned by the supplier in sales table
select * from supplier_request_product;

-- if quantity mentioned by supplier is present in product_inventory then it is processed in sales
-- -----------------------------------------------------
-- 6. Sales table to place the order
-- -----------------------------------------------------
call sales_BEFORE_INSERT(); -- checks whether quantity is zero or not 
select * from sales;
call sales_AFTER_INSERT(); -- reduces the quantity in product table and vales are inserted in order table i.e order is placed

-- -----------------------------------------------------
-- 7. Order table to confirm the sales of particular product
-- -----------------------------------------------------
select * from `order`;

 -- once the order is placed then total price for that supplier is calculated 
 -- -----------------------------------------------------
-- 8. Supplier_payment to order the product
-- -----------------------------------------------------
call sales_AFTER_INSERT(); -- to insert values in supplier_payment by considering quantity and price for total_price 
select * from supplier_payment;

-- inserting amount paid for that supplier which calculates the pending_amount
 -- -----------------------------------------------------
-- 9. Supplier_invoice to generate the bill statement
-- -----------------------------------------------------
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values('2017-12-13',900,1,1);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values('2017-12-17',800,1,2);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values('2017-08-13',900,2,3);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values('2017-10-30',1600,2,2);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values(now(),6000,3,4);
insert into supplier_invoice(invoice_date,Amount_paid,supplier_details_supplier_id,product_p_id) values(now(),6000,5,3);

select * from supplier_invoice;
call supplier_invoice_BEFORE_INSERT(); -- it calculates the pending_amount and updates the status
call Update_Pending_amount(); -- it is stored procedure to update the amount_paid
call supplier_invoice_BEFORE_UPDATE(); -- it updates the value inserted in stored procedure and updates the invoice table accordingly.

-- -----------------------------------------------------
-- View `inventory_system`.`highest_sold_product`
-- -----------------------------------------------------
select * from highest_sold_product;

-- -----------------------------------------------------
-- View `inventory_system`.`outstanding _amount`
-- -----------------------------------------------------
select * from outstanding _amount;


-- -----------------------------------------------------
-- To display the orders of all the suppliers
-- -----------------------------------------------------
select order_id,sp.supplier_id from `order` o 
inner join supplier_payment sp on o.supplier_id = sp.supplier_id 
inner join product p on sp.p_id = p.p_id 
group by sp.p_id;

-- -----------------------------------------------------
-- function calculate_loss
-- -----------------------------------------------------
select inventory_system.calculate_loss(1);
select inventory_system.calculate_loss(3);

-- -----------------------------------------------------
-- function calculate_profit
-- -----------------------------------------------------

select inventory_system.calculate_profit(2);
select inventory_system.calculate_profit(4);

-- -----------------------------------------------------
-- function supplier_product_amount
-- -----------------------------------------------------
select inventory_system.supplier_product_amount(1);

-- -----------------------------------------------------
-- procedure Update_Pending_amount
-- -----------------------------------------------------
call inventory_system.Update_Pending_amount(3, 5);

-- -----------------------------------------------------
-- procedure Products_brought_by_supplier
-- -----------------------------------------------------
call inventory_system.Products_brought_by_supplier(1);
