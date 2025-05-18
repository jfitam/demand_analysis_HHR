CREATE OR REPLACE FUNCTION analytics.metrics_class_by_dates(date_from date, date_to date)
returns setof analytics.metrics_class as $$
begin 
return query
 
	select 
		occ_class.train_departure_date_short::date, 
		occ_class.train_number::varchar(5), 
		tdt.departure_time::time as train_time, 
		occ_class."class"::text, 
		occ_class.corridor_name::text, 
		occ_class.direction::text, 
		occ_class.route_km::numeric, 
		occ_class.train_week_day::text, 
		occ_class.train_week_num::integer, 
		occ_class.train_year::integer,   --col10
		occ_class.total_occupancy::numeric, 
		occ_class.minimum_tickets::numeric,
		occ_class.minimum_tickets_promo::numeric, 
		occ_class.intermediate_tickets::numeric, 
		occ_class.intermediate_tickets_promo::numeric, 
		occ_class.maximum_tickets::numeric, 
		occ_class.maximum_tickets_promo::numeric, 
		occ_class.no_shown::numeric, 
		occ_class.residents::numeric, 
		occ_class.non_residents::numeric, --col20
		occ_class.individuals::numeric, 
		occ_class.groups::numeric, 
		occ_class.total_base_price::numeric, 
		occ_class.total_fees::numeric, 
		occ_class.total_penalties::numeric, 
		occ_class.total_amount_wo_vat::numeric, 
		occ_class.total_amount_compensated::numeric, 
		occ_class.available_seats::numeric, 
		occ_class.passengers_km::numeric, 
		occ_class.composition::numeric, --col30
		occ_class.comparable_day::date,
		
		-- available seats multiply per the route km 
		occ_class.available_seats::numeric * occ_class.route_km::numeric AS seats_km,
		
		occ_class.minimum_revenue::numeric, 
		occ_class.intermediate_revenue::numeric, 
		occ_class.maximum_revenue::numeric, 
		occ_class.no_level_revenue::numeric,
		occ_class.minimum_standard_price::numeric, 
		occ_class.intermediate_standard_price::numeric, 
		occ_class.maximum_standard_price::numeric, 
		occ_class.no_level_standard_price::numeric, --col40
		occ_class.tickets_adult::integer, 
		occ_class.tickets_children::integer, 
		occ_class.tickets_infant::integer, 
		occ_class.tickets_carer_adult::integer, 
		occ_class.tickets_carer_children::integer, 
		occ_class.tickets_prm_adult::integer, 
		occ_class.tickets_prm_children::integer,
		
		--sales channels
		occ_class.tickets_app_android::integer,
		occ_class.tickets_app_ios::integer,
		occ_class.tickets_app_ccs::integer,  --col50
		occ_class.tickets_app_crm::integer,
		occ_class.tickets_app_ota::integer,
		occ_class.tickets_app_tom::integer,
		occ_class.tickets_app_tvm::integer,
		occ_class.tickets_app_web::integer,

		--purchase anticipation
		occ_class.tickets_less_one_hour::integer,
		occ_class.tickets_between_1_and_6_hours::integer,
		occ_class.tickets_between_6_and_12_hours::integer,
		occ_class.tickets_between_12_and_24_hours::integer,
		occ_class.tickets_between_24_and_48_hours::integer,  --col60
		occ_class.tickets_between_48_and_72_hours::integer,
		occ_class.tickets_between_72_hours_and_1_week::integer,
		occ_class.tickets_without_operation_date_time::integer,
		occ_class.tickets_between_1_and_2_weeks::integer,
		occ_class.tickets_more_than_2_weeks::integer,
		
		--nationalities
		occ_class.tickets_saudi::integer,
		occ_class.tickets_egyptian::integer,
		occ_class.tickets_pakistani::integer,
		occ_class.tickets_indian::integer,
		occ_class.tickets_yemeni::integer, --col70
		occ_class.tickets_indonesian::integer,
		occ_class.tickets_jordanian::integer,
		occ_class.tickets_USA::integer,
		occ_class.tickets_britain::integer,
		
		--category of the day (added later)
		occ_class.day_category::varchar,
		
		--quotas
		occ_class.quota_long_route::integer,
		occ_class.quota_short_routes::integer, --col77
		occ_class.tickets_long_route::integer,
		occ_class.tickets_short_routes::integer
		
	from (
	-- this level is grouping from od to class
	 SELECT occ_od_km.train_departure_date_short,
		occ_od_km.train_number,
		occ_od_km.train_time,
		occ_od_km."class",
		occ_od_km.corridor_name,
		occ_od_km.direction,
		occ_od_km.route_km,
		to_char(occ_od_km.train_departure_date_short::timestamp with time zone, 'Dy'::text) AS train_week_day,
		occ_od_km.train_week_num,
		occ_od_km.train_year,
		
		-- adding the occupancy for all the ods
		sum(occ_od_km.occupancy) AS total_occupancy,
		sum(occ_od_km.minimum_tickets) AS minimum_tickets,
		sum(occ_od_km.minimum_tickets_promo) AS minimum_tickets_promo,
		sum(occ_od_km.intermediate_tickets) AS intermediate_tickets,
		sum(occ_od_km.intermediate_tickets_promo) AS intermediate_tickets_promo,
		sum(occ_od_km.maximum_tickets) AS maximum_tickets,
		sum(occ_od_km.maximum_tickets_promo) AS maximum_tickets_promo,
		sum(occ_od_km.no_shown) AS no_shown,
		sum(occ_od_km.residents) AS residents,
		sum(occ_od_km.non_residents) AS non_residents,
		sum(occ_od_km.individuals) AS individuals,
		sum(occ_od_km.groups) AS groups,
		
		--adding the income for all the ods, separating base prices, the fees, and the penalties
		sum(occ_od_km.base_price) as total_base_price,
		sum(occ_od_km.fees) as total_fees,
		sum(occ_od_km.penalties) as total_penalties,
		sum(occ_od_km.base_price+occ_od_km.fees+occ_od_km.penalties) as total_amount_wo_vat,
		
		--compensations
		sum(occ_od_km.amount_compensated) as total_amount_compensated,
		
		-- select the number of seats according to the class. keep null in case the class it is not recognized (there is an error with the entry)
		max(occ_od_km.composition) * 
		case
			when occ_od_km."class"='Economy' then 304
			when occ_od_km."class"='Business' then 113
			else NULL
		end AS available_seats,  
		
		-- add each od pax*km	
		sum(occ_od_km.occupancy_without_infants::numeric * occ_od_km.od_km) AS passengers_km, 
	
		--the composition of the class
		max(occ_od_km.composition) as composition,
		
		--the day to compare
		occ_od_km.comparable_day,
		occ_od_km.day_category,
		
		-- adding the information about revenue per levels
		sum(occ_od_km.minimum_revenue) as minimum_revenue,
		sum(occ_od_km.intermediate_revenue) as intermediate_revenue,
		sum(occ_od_km.maximum_revenue) as maximum_revenue,
		sum(occ_od_km.no_level_revenue) as no_level_revenue,
		sum(occ_od_km.minimum_standard_price) as minimum_standard_price,
		sum(occ_od_km.intermediate_standard_price) as intermediate_standard_price,
		sum(occ_od_km.maximum_standard_price) as maximum_standard_price,
		sum(occ_od_km.no_level_standard_price) as no_level_standard_price,
		
		--tickets per profile
		sum(occ_od_km.tickets_adult)::integer as tickets_adult,
		sum(occ_od_km.tickets_children)::integer as tickets_children,
		sum(occ_od_km.tickets_infant)::integer as tickets_infant,
		sum(occ_od_km.tickets_carer_adult)::integer as tickets_carer_adult,
		sum(occ_od_km.tickets_carer_children)::integer as tickets_carer_children,
		sum(occ_od_km.tickets_prm_adult)::integer as tickets_prm_adult,
		sum(occ_od_km.tickets_prm_children)::integer as tickets_prm_children,
		
		--sales channels
		sum(occ_od_km.tickets_app_android)::integer as tickets_app_android,
		sum(occ_od_km.tickets_app_ios)::integer as tickets_app_ios,
		sum(occ_od_km.tickets_app_ccs)::integer as tickets_app_ccs,
		sum(occ_od_km.tickets_app_crm)::integer as tickets_app_crm,
		sum(occ_od_km.tickets_app_ota)::integer as tickets_app_ota,
		sum(occ_od_km.tickets_app_tom)::integer as tickets_app_tom,
		sum(occ_od_km.tickets_app_tvm)::integer as tickets_app_tvm,
		sum(occ_od_km.tickets_app_web)::integer as tickets_app_web,

		--purchase anticipation
		
		sum(occ_od_km.tickets_less_one_hour)::integer as tickets_less_one_hour,
		sum(occ_od_km.tickets_between_1_and_6_hours)::integer as tickets_between_1_and_6_hours,
		sum(occ_od_km.tickets_between_6_and_12_hours)::integer as tickets_between_6_and_12_hours,
		sum(occ_od_km.tickets_between_12_and_24_hours)::integer as tickets_between_12_and_24_hours,
		sum(occ_od_km.tickets_between_24_and_48_hours)::integer as tickets_between_24_and_48_hours,
		sum(occ_od_km.tickets_between_48_and_72_hours)::integer as tickets_between_48_and_72_hours,
		sum(occ_od_km.tickets_between_72_hours_and_1_week)::integer as tickets_between_72_hours_and_1_week,
		sum(occ_od_km.tickets_without_operation_date_time)::integer as tickets_without_operation_date_time,
		sum(occ_od_km.tickets_between_1_and_2_weeks)::integer as tickets_between_1_and_2_weeks,
		sum(occ_od_km.tickets_more_than_2_weeks)::integer as tickets_more_than_2_weeks,
		
		-- nationalities
		sum(occ_od_km.tickets_saudi)::integer as tickets_saudi,
		sum(occ_od_km.tickets_egyptian)::integer as tickets_egyptian,
		sum(occ_od_km.tickets_pakistani)::integer as tickets_pakistani,
		sum(occ_od_km.tickets_indian)::integer as tickets_indian,
		sum(occ_od_km.tickets_yemeni)::integer as tickets_yemeni,
		sum(occ_od_km.tickets_indonesian)::integer as tickets_indonesian,
		sum(occ_od_km.tickets_jordanian)::integer as tickets_jordanian,
		sum(occ_od_km.tickets_USA)::integer as tickets_USA,
		sum(occ_od_km.tickets_britain)::integer as tickets_britain,
		
		-- adding quota information
		max(occ_od_km.quota) FILTER (WHERE  occ_od_km.od_km = occ_od_km.route_km) as quota_long_route,
		max(occ_od_km.quota) FILTER (WHERE  occ_od_km.od_km < occ_od_km.route_km) as quota_short_routes,
		max(occ_od_km.occupancy_without_infants) FILTER (WHERE  occ_od_km.od_km = occ_od_km.route_km) as tickets_long_route,
		max(occ_od_km.occupancy_without_infants) FILTER (WHERE  occ_od_km.od_km < occ_od_km.route_km) as tickets_short_routes
		
		
	   FROM 
	   ( SELECT train_offer.train_departure_date_short,
				train_offer.od_departure_date_short,
				train_offer.train_number,
				train_offer.train_time,
				classes."class",
				train_offer.composition,
				ticket_info.od,
				ck_route.corridor_name,
				ck_route.direction,
				ck_route.route_km,
				coalesce(train_w.week_num, 0) AS train_week_num,
				ticket_info.occupancy,
				ticket_info.occupancy_without_infants,
				ticket_info.minimum_tickets,
				ticket_info.minimum_tickets_promo,
				ticket_info.intermediate_tickets,
				ticket_info.intermediate_tickets_promo,
				ticket_info.maximum_tickets,
				ticket_info.maximum_tickets_promo,
				ticket_info.no_shown,
				ticket_info.residents,
				ticket_info.non_residents,
				ticket_info.groups,
				ticket_info.individuals,
				
				--quotas
				occupancy.quota,
				
				--deprecated
				--CASE 
				--	WHEN ticket_info.occupancy_without_infants = 0 THEN 0
				--	ELSE ticket_info.occupancy_without_infants / occupancy.quota
				--END AS quota_load,
				
				
				-- financial
				ticket_info.base_price,
				ticket_info.fees,
				ticket_info.penalties,
				ticket_info.minimum_revenue,
				ticket_info.intermediate_revenue,
				ticket_info.maximum_revenue,
				ticket_info.no_level_revenue,
				ticket_info.minimum_standard_price,
				ticket_info.intermediate_standard_price,
				ticket_info.maximum_standard_price,
				ticket_info.no_level_standard_price,

				--profiles
				ticket_info.tickets_adult,
				ticket_info.tickets_children,
				ticket_info.tickets_infant,
				ticket_info.tickets_carer_adult,
				ticket_info.tickets_carer_children,
				ticket_info.tickets_prm_adult,
				ticket_info.tickets_prm_children,
				
				--sales channels
				ticket_info.tickets_app_android,
				ticket_info.tickets_app_ios,
				ticket_info.tickets_app_ccs,
				ticket_info.tickets_app_crm,
				ticket_info.tickets_app_ota,
				ticket_info.tickets_app_tom,
				ticket_info.tickets_app_tvm,
				ticket_info.tickets_app_web,

				--purchase anticipation
				ticket_info.tickets_without_operation_date_time,
				ticket_info.tickets_less_one_hour,
				ticket_info.tickets_between_1_and_6_hours,
				ticket_info.tickets_between_6_and_12_hours,
				ticket_info.tickets_between_12_and_24_hours,
				ticket_info.tickets_between_24_and_48_hours,
				ticket_info.tickets_between_48_and_72_hours,
				ticket_info.tickets_between_72_hours_and_1_week,
				ticket_info.tickets_between_1_and_2_weeks,
				ticket_info.tickets_more_than_2_weeks,
				
				--nationalities
				ticket_info.tickets_saudi,
				ticket_info.tickets_egyptian,
				ticket_info.tickets_pakistani,
				ticket_info.tickets_indian,
				ticket_info.tickets_yemeni,
				ticket_info.tickets_indonesian,
				ticket_info.tickets_jordanian,
				ticket_info.tickets_USA,
				ticket_info.tickets_britain,
				
				ticket_info.amount_compensated,
				ck.od_km,
				cal.comparable_day,
				COALESCE(cal.category, 'No info') as day_category,
				coalesce(train_w.year,0) AS train_year
				
			   FROM 
				
				(-- get the offer per service from the trains that sold joining with the composition
					SELECT 
					COALESCE(trains_sold.train_departure_date_short, composition.train_date) as train_departure_date_short,
					COALESCE(LPAD(trains_sold.train_number,5,'0'), LPAD(composition.train_number,5,'0')) as train_number,
					composition.train_time,
					trains_sold.od,
					trains_sold.od_departure_date_short,
					composition.composition
					
					FROM (select train_list.train_departure_date_short, train_list.train_number, train_list.stretch,
					
					--added the od to make sure that the income appears even if there is no tickets in the train list for specific od
					train_list.od,
					
					--added the od departure to join with the booking payment detailed
					train_list.departure_date_short as od_departure_date_short

					FROM "AFC".train_list
					WHERE train_list.train_departure_date_short BETWEEN date_from AND date_to
					GROUP BY train_list.train_departure_date_short, train_list.train_number, train_list.stretch, train_list.od, train_list.departure_date_short) trains_sold
					
					-- add the composition
					FULL JOIN (SELECT train_date, train_number, train_time, max(composition) as composition from analytics.composition_full where train_date between date_from and date_to group by train_date, train_time, train_number) composition 
					
					 -- join composition
					ON composition.train_date = trains_sold.train_departure_date_short
					AND LPAD(composition.train_number,5,'0') = LPAD(trains_sold.train_number,5,'0') ) train_offer
				
				--add the classes
				CROSS JOIN (SELECT 'Economy' as "class" UNION SELECT 'Business' as "class") classes
				
				-- get the maximum km of the route and the name of the corridor
				 LEFT JOIN (select corridor_name, digit_corridor, direction, parity_digit, max(od_km) as route_km from "AFC".corridor_od_direction group by corridor_name, digit_corridor, direction, parity_digit) ck_route 
				 ON "substring"(train_offer.train_number::text, 2, 1)::integer = ck_route.digit_corridor 
				 AND ("substring"(train_offer.train_number::text, 5, 1)::integer % 2) = (ck_route.parity_digit::integer % 2)
				 
				 
				-- join with the train list without the infants, by OD
				LEFT JOIN (
					-- subquery to get information about tickets and income
					SELECT
					tl.train_departure_date_short,
					tl.train_number,
					tl."class",
					tl.od,
					tl.stretch,
					
					-- passenger information
					count(*) AS occupancy,
					count(*) FILTER (WHERE tl.profile <> 'Infant') AS occupancy_without_infants,
					count(*) FILTER (WHERE tl.validating_time IS NULL) as no_shown,
					count(*) FILTER (WHERE tl.price_level = 'Minimum' and tl.promo_price <> 'Y') as minimum_tickets,
					count(*) FILTER (WHERE tl.price_level = 'Minimum' and tl.promo_price = 'Y') as minimum_tickets_promo,
					count(*) FILTER (WHERE tl.price_level = 'Intermediate' and tl.promo_price <> 'Y') as intermediate_tickets,
					count(*) FILTER (WHERE tl.price_level = 'Intermediate' and tl.promo_price = 'Y') as intermediate_tickets_promo,
					count(*) FILTER (WHERE tl.price_level = 'Maximum' and tl.promo_price <> 'Y') as maximum_tickets,
					count(*) FILTER (WHERE tl.price_level = 'Maximum' and tl.promo_price = 'Y') as maximum_tickets_promo,
					count(*) FILTER (WHERE bpd.residency <> 'Resident') as non_residents,
					count(*) FILTER (WHERE bpd.residency = 'Resident') as residents,
					count(*) FILTER (WHERE tl.groupyn = 'Yes' AND tl.sales_channel <> 'OTA') as groups,
					count(*) FILTER (WHERE tl.groupyn <> 'Yes' OR tl.sales_channel = 'OTA' and tl.profile <> 'Infant') as individuals, --remove the infants to match the groups availability
					
					
					-- finantial information
					-- the amount will be the total including fees but without VAT. The penalty tariff has to be added if want to include the penalties for cancelling
					sum(tl.base_price) as base_price,
					sum(tl.management_fee + tl.payment_fee) as fees,
					sum(tl.penalty_tariff) as penalties,
					sum(bpd.amount_compensated) as amount_compensated,
					sum(tl.base_price+tl.management_fee + tl.payment_fee) FILTER (WHERE tl.price_level = 'Minimum') as minimum_revenue,
					sum(tl.base_price+tl.management_fee + tl.payment_fee) FILTER (WHERE tl.price_level = 'Intermediate') as intermediate_revenue,
					sum(tl.base_price+tl.management_fee + tl.payment_fee) FILTER (WHERE tl.price_level = 'Maximum') as maximum_revenue,
					sum(tl.base_price+tl.management_fee + tl.payment_fee) FILTER (WHERE tl.price_level IS NULL OR (tl.price_level <> 'Maximum' and tl.price_level <> 'Intermediate' and tl.price_level <> 'Minimum')) as no_level_revenue,
					sum(CASE WHEN (tl.standard_price* tl.discount) < tl.base_price THEN tl.base_price ELSE tl.standard_price* tl.discount END  + tl.management_fee + tl.payment_fee) FILTER (WHERE tl.price_level = 'Minimum') as minimum_standard_price,
					sum(CASE WHEN (tl.standard_price* tl.discount) < tl.base_price THEN tl.base_price ELSE tl.standard_price* tl.discount END  +tl.management_fee + tl.payment_fee) FILTER (WHERE tl.price_level = 'Intermediate') as intermediate_standard_price,
					sum(CASE WHEN (tl.standard_price* tl.discount) < tl.base_price THEN tl.base_price ELSE tl.standard_price* tl.discount END  +tl.management_fee + tl.payment_fee) FILTER (WHERE tl.price_level = 'Maximum') as maximum_standard_price,
					sum(COALESCE(CASE WHEN (tl.standard_price* tl.discount) < tl.base_price THEN tl.base_price ELSE tl.standard_price* tl.discount END  +tl.management_fee + tl.payment_fee, tl.base_price+tl.management_fee + tl.payment_fee)) FILTER (WHERE tl.price_level IS NULL OR (tl.price_level <> 'Maximum' and tl.price_level <> 'Intermediate' and tl.price_level <> 'Minimum')) as no_level_standard_price,
					
					--profiles
					count(*) FILTER (WHERE tl.profile = 'Adult') AS tickets_adult,
					count(*) FILTER (WHERE tl.profile = 'Children') AS tickets_children,
					count(*) FILTER (WHERE tl.profile = 'Infant') AS tickets_infant,
					count(*) FILTER (WHERE tl.profile = 'Carer Adult') AS tickets_carer_adult,
					count(*) FILTER (WHERE tl.profile = 'Carer Children') AS tickets_carer_children,
					count(*) FILTER (WHERE tl.profile = 'PRM Adult') AS tickets_prm_adult,
					count(*) FILTER (WHERE tl.profile = 'PRM Children') AS tickets_prm_children,
					
					--sales channels
					count(*) FILTER (WHERE tl.sales_channel = 'APP_A') AS tickets_app_android,
					count(*) FILTER (WHERE tl.sales_channel = 'APP_I') AS tickets_app_ios,
					count(*) FILTER (WHERE tl.sales_channel = 'CCS') AS tickets_app_ccs,
					count(*) FILTER (WHERE tl.sales_channel = 'CRM') AS tickets_app_crm,
					count(*) FILTER (WHERE tl.sales_channel = 'OTA') AS tickets_app_ota,
					count(*) FILTER (WHERE tl.sales_channel = 'TOM') AS tickets_app_tom,
					count(*) FILTER (WHERE tl.sales_channel = 'TVM') AS tickets_app_tvm,
					count(*) FILTER (WHERE tl.sales_channel = 'WEB') AS tickets_app_web,
					
					--purchase anticipation
					COUNT(*) FILTER (where bpd.operation_date_time is NULL) AS tickets_without_operation_date_time,
					COUNT(*) FILTER (where bpd.operation_date_time is NOT NULL AND (tl.departure_date - bpd.operation_date_time) < '01:00:00') AS tickets_less_one_hour,
					
					COUNT(*) FILTER (
						WHERE bpd.operation_date_time is NOT NULL 
						AND (tl.departure_date - bpd.operation_date_time) < '06:00:00' 
						AND (tl.departure_date - bpd.operation_date_time) >= '01:00:00') AS tickets_between_1_and_6_hours,
						
					COUNT(*) FILTER (
						WHERE bpd.operation_date_time is NOT NULL 
						AND (tl.departure_date - bpd.operation_date_time) < '12:00:00' 
						AND (tl.departure_date - bpd.operation_date_time) >= '06:00:00') AS tickets_between_6_and_12_hours,
					
					COUNT(*) FILTER (
						WHERE bpd.operation_date_time is NOT NULL 
						AND (tl.departure_date - bpd.operation_date_time) < '24:00:00' 
						AND (tl.departure_date - bpd.operation_date_time) >= '12:00:00') AS tickets_between_12_and_24_hours,
						
					COUNT(*) FILTER (
						WHERE bpd.operation_date_time is NOT NULL 
						AND (tl.departure_date - bpd.operation_date_time) < '48:00:00' 
						AND (tl.departure_date - bpd.operation_date_time) >= '24:00:00') AS tickets_between_24_and_48_hours,

					COUNT(*) FILTER (
						WHERE bpd.operation_date_time is NOT NULL 
						AND (tl.departure_date - bpd.operation_date_time) < '72:00:00' 
						AND (tl.departure_date - bpd.operation_date_time) >= '48:00:00') AS tickets_between_48_and_72_hours,
					
					COUNT(*) FILTER (
						WHERE bpd.operation_date_time is NOT NULL 
						AND (tl.departure_date - bpd.operation_date_time) < '168:00:00' 
						AND (tl.departure_date - bpd.operation_date_time) >= '72:00:00') AS tickets_between_72_hours_and_1_week,

					COUNT(*) FILTER (
						WHERE bpd.operation_date_time is NOT NULL 
						AND (tl.departure_date - bpd.operation_date_time) < '336:00:00' 
						AND (tl.departure_date - bpd.operation_date_time) >= '168:00:00') AS tickets_between_1_and_2_weeks,
						
					COUNT(*) FILTER (
						WHERE bpd.operation_date_time is NOT NULL 
						AND (tl.departure_date - bpd.operation_date_time) >= '336:00:00') AS tickets_more_than_2_weeks,
						
					-- nationalities
					count(*) FILTER (WHERE tl.nationality = 'SA') AS tickets_saudi,
					count(*) FILTER (WHERE tl.nationality = 'EG') AS tickets_egyptian,
					count(*) FILTER (WHERE tl.nationality = 'PK') AS tickets_pakistani,
					count(*) FILTER (WHERE tl.nationality = 'IN') AS tickets_indian,
					count(*) FILTER (WHERE tl.nationality = 'YE') AS tickets_yemeni,
					count(*) FILTER (WHERE tl.nationality = 'ID') AS tickets_indonesian,
					count(*) FILTER (WHERE tl.nationality = 'JO') AS tickets_jordanian,
					count(*) FILTER (WHERE tl.nationality = 'US') AS tickets_USA,
					count(*) FILTER (WHERE tl.nationality = 'GB') AS tickets_britain

	
					
					FROM ( 
						-- get the train list by ticket number
						SELECT train_list.train_departure_date_short,
						train_list.departure_date_short as od_departure_date_short,
						train_list.departure_date,
						train_list.train_number,
						train_list."class",
						train_list.od,
						train_list.profile,
						train_list.sales_channel,
						train_list.stretch,
						train_list.validating_time,
						train_list.nationality,
						fares.price_level,
						COALESCE(fares.promo_price,'N') as promo_price,
						train_list.ticket_number,
						COALESCE(train_list.groupyn, 'No') as groupyn,
						
						--standard price
						max(fares_prices.price) as standard_price,
						max(profiles.discount) as discount,
						
						--finantial information
						max(train_list.base_price) as base_price,
						max(train_list.management_fee) as management_fee,
						max(train_list.payment_fee) as payment_fee,
						max(train_list.penalty_tariff) as penalty_tariff
						
					   FROM "AFC".train_list
					   
					   --add the price levels
					   LEFT JOIN "AFC".fares ON train_list.tariff = fares.tariff
					   
					   --add the standard prices
					   LEFT JOIN "AFC".fares_prices 
						ON fares_prices.price_set = 'A' 
						and train_list.class = fares_prices.class
						and train_list.od = fares_prices.od
						and fares.price_level = fares_prices.price_level
						and train_list.train_departure_date_short between fares_prices.trip_date_from and fares_prices.trip_date_to

					   --add the discount according to the profile					
					   LEFT JOIN "AFC".profiles ON train_list.profile = profiles.profile and train_list."class" = profiles."class"
					   
					   -- filter infants and trains that hasn't circulated yet
					  WHERE train_departure_date_short between date_from and date_to -- and train_list.profile::text <> 'Infant'::text
					  
					  -- filter also compensation for Canceled Trains (which actually means those tickets did not travelled
					  AND coalesce(compensation_type,'') <> 'Canceled Train'
					  
					  GROUP BY 
						train_list.train_departure_date_short,
						train_list.departure_date,
						train_list.departure_date_short,
						train_list.ticket_number, 
						train_list.validating_time,
						train_list.nationality,
						train_list.train_number, 
						train_list.od, 
						train_list.profile,
						train_list.sales_channel,
						train_list."class", 
						train_list.stretch,
						COALESCE(train_list.groupyn, 'No'),
						fares.price_level,
						COALESCE(fares.promo_price,'N')) tl
					  	
						-- get the income (by ticket number) from the booking payment detailed and join to the previous 
						LEFT JOIN (
							SELECT 
								date(bpd.departure_date_time) as od_departure_date_short,
								min(operation_date_time) as operation_date_time,
								bpd.train_number, bpd.od, bpd."class", 
								COALESCE(rc.residency, 'Not Resident') as residency,
								bpd.ticket_number,
								-- the amount will be the total including fees but without VAT. The penalty tariff has to be added if want to include the penalties for cancelling
								sum(bpd.base_price + bpd.management_fee + bpd.payment_fee) FILTER (WHERE bpd.compensation_status = 'Refunded')  as amount_compensated

							
							FROM "AFC".booking_payment_detailed bpd
							
							-- get the residency from the table of the criterion
							LEFT JOIN analytics.residency_criterion rc on bpd.document_type = rc.document_type
							
							WHERE date(bpd.departure_date_time) BETWEEN date_from AND date_to
							
							GROUP BY 
								date(bpd.departure_date_time),
								bpd.train_number, 
								bpd.od, 
								bpd."class", 
								bpd.ticket_number, 
								rc.residency
						
						) bpd
						
						-- connection with the train list and the booking payment detailed
						on tl.ticket_number = bpd.ticket_number
						and tl.od_departure_date_short = bpd.od_departure_date_short 
						and tl.train_number = bpd.train_number 
						and tl.od = bpd.od 
						and tl."class" = bpd."class"
					
					--group by the result of train list and bookng payment detail in the same level of grouping as the rest of the tables
					GROUP BY  
						tl.train_departure_date_short,
						tl.train_number,
						tl."class",
						tl.stretch,
						tl.od ) ticket_info
					
					--connection between the train list/bpd and the train offer with class
					  ON train_offer.train_departure_date_short = ticket_info.train_departure_date_short
					  AND train_offer.train_number = ticket_info.train_number
					  AND classes."class" = ticket_info."class"
					  AND train_offer.od = ticket_info.od
					
					-- join with the corridor to get the km of the od
					LEFT JOIN "AFC".corridor_od_direction ck ON "substring"(train_offer.train_number::text, 2, 1)::integer = ck.digit_corridor AND train_offer.od::text = ck.od_name AND ("substring"(train_offer.train_number::text, 5, 1)::integer % 2) = (ck.parity_digit::integer % 2)
					 
					-- get the number of week from the relation table, not from a function
					LEFT JOIN "AFC".weeks train_w ON train_offer.train_departure_date_short >= train_w.date_from AND train_offer.train_departure_date_short <= train_w.date_to
					
					-- get the comparable day
					LEFT JOIN analytics.calendar cal ON cal.day = train_offer.train_departure_date_short

					-- get the quotas of each od from occupancy
					LEFT JOIN (
						SELECT DISTINCT ON (date, od, train_number, class) date, od, train_number, class, quota_configuration as quota
						FROM "AFC".occupancy_list_hist
						WHERE date between date_from AND date_to + INTERVAL '1 day' and quota_configuration > 0
						ORDER BY date, od, train_number, class, data_date desc, quota_configuration DESC
					) occupancy 
					ON occupancy.date = train_offer.od_departure_date_short AND occupancy.class = classes.class AND occupancy.od = ticket_info.od AND occupancy.train_number = train_offer.train_number
				
				-- end of the first query
				) occ_od_km
				

				 
	  -- at the end the grouping is by train and class
	  GROUP BY occ_od_km.train_departure_date_short, occ_od_km.train_time, occ_od_km.train_number, occ_od_km."class", occ_od_km.corridor_name, occ_od_km.direction, occ_od_km.train_week_num, occ_od_km.train_year, occ_od_km.route_km, occ_od_km.comparable_day, occ_od_km.day_category
  	) occ_class
	
	LEFT JOIN "AFC".train_departure_times tdt ON tdt.train_number = occ_class.train_number;
	
	
	
	
end;
$$ language plpgsql;