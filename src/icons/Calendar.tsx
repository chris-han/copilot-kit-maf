import { Calendar as LucideCalendar, LucideProps } from 'lucide-react';

const Calendar = ({ className, ...props }: LucideProps) => {
  return <LucideCalendar className={className} {...props} />;
};

export default Calendar;